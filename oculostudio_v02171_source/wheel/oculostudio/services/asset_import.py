"""Background import, with progress and immutable browser-upload storage.

A native picker/path import references the source in place. Browser file input
and drag/drop deliver File bytes, not local paths, and use bounded chunks into
project/imports. No byte buffers or uploads leave the local Studio server.
"""
from __future__ import annotations
import hashlib
import os
import re
import shutil
import stat
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from oculostudio.errors import ValidationError
from oculostudio.services.assets import add_asset
from oculostudio.util import CancellationToken

CHUNK_BYTES = 8 * 1024 * 1024
RESERVE_BYTES = 64 * 1024 * 1024


def _validate_name(name: str) -> str:
    if (not name or name in {'.','..'} or len(name) > 240
        or any(c in name for c in '/\\:\x00<>|?*"') or name.endswith((' ','.'))
        or any(ord(c) < 32 for c in name)):
        raise ValidationError('Invalid upload filename; choose an ordinary file, not a path')
    if name.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}:
        raise ValidationError('Reserved Windows filename')
    return name


def normalize_local_path(value: str) -> Path:
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    if not value or 'fakepath' in value.lower():
        raise ValidationError('This is not a real local path. Use Choose files or drop the file instead.')
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise ValidationError(f'File or folder does not exist: {path}')
    if not path.is_file() and not path.is_dir():
        raise ValidationError('Only ordinary files and recording folders are supported')
    return path


def check_zip_structure(path: Path):
    if path.suffix.lower() != '.zip':
        return
    try:
        with zipfile.ZipFile(path) as archive:
            if not archive.infolist():
                raise ValidationError('The ZIP archive is empty')
            for info in archive.infolist():
                name = info.filename.replace('\\','/')
                part = PurePosixPath(name)
                if (part.is_absolute() or '..' in part.parts or '\x00' in name
                    or (part.parts and ':' in part.parts[0])):
                    raise ValidationError(f'Unsafe ZIP member path: {info.filename}')
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValidationError('Symbolic links in ZIP archives are not supported')
                if info.flag_bits & 1:
                    raise ValidationError('Password-protected ZIP archives are not supported')
    except zipfile.BadZipFile as exc:
        raise ValidationError('Cannot read ZIP directory: the file may be incomplete or corrupt') from exc


def _existing(store, sid, path, participant_id, metadata_extra=None):
    snap = store.snapshot()
    requested = metadata_extra or {}
    requested_scope = str(requested.get("asset_scope") or "interaction")
    requested_set = str(requested.get("calibration_set_id") or "")
    for a in snap.sessions[sid].assets.values():
        existing_scope = str((a.metadata or {}).get("asset_scope") or "interaction")
        existing_set = str((a.metadata or {}).get("calibration_set_id") or "")
        if (os.path.normcase(str(Path(a.source_path).resolve())) == os.path.normcase(str(path.resolve()))
            and a.participant_id == participant_id
            and existing_scope == requested_scope
            and (requested_scope != "calibration" or existing_set == requested_set)):
            return a
    return None


def import_local_asset(store, *, session_id, path, label=None, participant_id=None, checksum=True,
                       storage_mode='local_reference', original_name=None, known_sha256=None,
                       metadata_extra=None, progress=None, cancellation=None):
    token = cancellation or CancellationToken.create()
    token.raise_if_cancelled()
    path = normalize_local_path(path)
    if session_id not in store.snapshot().sessions:
        raise ValidationError('Import session no longer exists')
    existing = _existing(store, session_id, path, participant_id, metadata_extra=metadata_extra)
    if existing:
        return {'asset':existing.model_dump(mode='json'), 'reused':True}
    if progress: progress(0.02, f'Checking {path.name}')
    check_zip_structure(path)
    before = path.stat() if path.is_file() else None
    digest = known_sha256
    if checksum and path.is_file() and not digest:
        h = hashlib.sha256(); done = 0
        last_report = 0.0
        with path.open('rb') as stream:
            while True:
                token.raise_if_cancelled()
                chunk = stream.read(CHUNK_BYTES)
                if not chunk: break
                h.update(chunk); done += len(chunk)
                now = time.monotonic()
                if progress and (now-last_report > 0.2 or done == before.st_size):
                    progress(0.05+0.75*done/max(1,before.st_size), f'Checking SHA-256: {done:,} / {before.st_size:,} bytes')
                    last_report = now
        digest = h.hexdigest()
    token.raise_if_cancelled()
    if progress: progress(0.85, 'Inspecting recording format and timestamps')
    # Probe happens before the atomic mutation inside add_asset. The hook stops
    # cancellation from publishing a half-finished registration.
    def before_register():
        token.raise_if_cancelled()
        if before is not None:
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValidationError('Source changed while importing; try again after the write finishes')
    a = add_asset(store, session_id=session_id, path=path, label=label,
                  participant_id=participant_id, checksum=False,
                  source_sha256_override=digest, before_register=before_register,
                  metadata_extra={**(metadata_extra or {}), 'import': {'storage_mode':storage_mode, 'original_filename':original_name or path.name,
                                            'checksum_scope':'whole_file' if digest else 'not_computed'}})
    if progress: progress(1.0, f'Registered {a.label} ({a.kind})')
    return {'asset':a.model_dump(mode='json'), 'reused':False, 'storage_mode':storage_mode}


@dataclass
class Upload:
    id: str
    store: object
    session_id: str
    name: str
    size: int
    stage_dir: Path
    scope: str = "interaction"
    calibration_set_id: str | None = None
    count: int = 0
    sha: object = field(default_factory=hashlib.sha256)
    state: str = 'receiving'
    lock: object = field(default_factory=threading.RLock)

    @property
    def data(self): return self.stage_dir/'payload.part'


class UploadManager:
    def __init__(self):
        self._items = {}; self._lock = threading.RLock()

    def start(self, store, *, session_id, filename, size, root=None, scope="interaction", calibration_set_id=None):
        _validate_name(filename)
        if size <= 0:
            raise ValidationError('Empty files cannot be imported; select the original ZIP')
        if session_id not in store.snapshot().sessions:
            raise ValidationError('Unknown import session')
        root = Path(root) if root is not None else store.import_dir(session_id)
        root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            reserved = sum(max(0,u.size-u.count) for u in self._items.values() if u.store.root == store.root and u.state == 'receiving')
            if shutil.disk_usage(root).free < size + reserved + RESERVE_BYTES:
                raise ValidationError('Not enough project disk space for a local copy. Use the native file picker or paste the original path to reference it instead.')
            uid = uuid.uuid4().hex
            directory = root/'.incoming'/uid
            directory.mkdir(parents=True, exist_ok=False)
            item = Upload(uid, store, session_id, filename, int(size), directory, scope=str(scope or "interaction"), calibration_set_id=str(calibration_set_id or "") or None)
            item.data.touch(exist_ok=False)
            self._items[uid] = item
        return {'upload_id':uid, 'chunk_bytes':CHUNK_BYTES, 'offset':0, 'storage_mode':'project_copy'}

    def get(self, store, upload_id):
        with self._lock:
            item = self._items.get(upload_id)
        if item is None:
            raise ValidationError('Upload not found; please select the file again')
        if item.store is not store:
            raise ValidationError('Project changed during import; switch back or start a new import')
        return item

    def append(self, store, upload_id, offset, data):
        item = self.get(store, upload_id)
        with item.lock:
            if item.state != 'receiving': raise ValidationError('Upload is no longer receiving data')
            if offset != item.count: raise ValidationError(f'Unexpected chunk offset: expected {item.count}, got {offset}')
            if not data or len(data)>CHUNK_BYTES or item.count+len(data)>item.size:
                raise ValidationError('Invalid upload chunk size')
            token_start = item.count
            try:
                with item.data.open('ab') as stream: stream.write(data)
            except OSError:
                with item.data.open('r+b') as stream: stream.truncate(token_start)
                raise
            item.sha.update(data); item.count+=len(data)
            return {'upload_id':item.id, 'offset':item.count, 'size':item.size}

    def seal(self, store, upload_id):
        item = self.get(store, upload_id)
        with item.lock:
            if item.state != 'receiving': raise ValidationError('Upload was already submitted')
            if item.count != item.size: raise ValidationError('Upload is incomplete; no asset was registered')
            item.state = 'sealed'
        return item

    def abandon(self, store, upload_id):
        item = self.get(store, upload_id)
        with item.lock:
            if item.state == 'processing':
                raise ValidationError('Cancel the import job while inspection is running')
            if item.state == 'registered':
                raise ValidationError('Import is complete; removing a source reference never deletes its file')
            item.state='cancelled'
            shutil.rmtree(item.stage_dir, ignore_errors=True)
        with self._lock: self._items.pop(upload_id,None)

    def finish_job(self, item, *, label=None, participant_id=None, progress=None, cancellation=None):
        token = cancellation or CancellationToken.create()
        destination = item.store.import_dir(item.session_id)/item.id/item.name
        moved = False
        try:
            with item.lock:
                token.raise_if_cancelled()
                if item.state != 'sealed': raise ValidationError('Upload was cancelled or already finalized')
                item.state = 'processing'
                destination.parent.mkdir(parents=True,exist_ok=False)
                item.data.replace(destination); moved=True
            result = import_local_asset(item.store, session_id=item.session_id, path=destination,
                label=label or item.name, participant_id=participant_id, checksum=True,
                known_sha256=item.sha.hexdigest(), storage_mode='project_copy', original_name=item.name,
                progress=progress, cancellation=token)
            item.state='registered'
            return result
        finally:
            # Never delete any path already registered, including a successful
            # registration just before an external cancellation request.
            if moved and _existing(item.store,item.session_id,destination,participant_id) is None:
                shutil.rmtree(destination.parent,ignore_errors=True)
            shutil.rmtree(item.stage_dir,ignore_errors=True)
            with self._lock: self._items.pop(item.id,None)

    def finish_calibration_job(self, item, *, calibration_set_id, label=None, participant_id=None,
                               progress=None, cancellation=None):
        """Finalize a browser/drop upload directly into a Project Calibration Set.

        Browser File objects do not expose local paths. Their bytes are streamed in
        bounded chunks into the local project, then registered as project-level
        calibration evidence without ever entering the Interaction asset scope.
        """
        from oculostudio.services.calibration_registry import (
            calibration_metadata, infer_calibration_role, infer_participant,
            set_asset_calibration_scope,
        )
        from oculostudio.services.calibration_sets import list_calibration_sets

        token = cancellation or CancellationToken.create()
        rows = list_calibration_sets(item.store)
        target = next((row for row in rows if row.get("id") == calibration_set_id or str(row.get("code", "")).casefold() == str(calibration_set_id).casefold()), None)
        if target is None:
            raise ValidationError(f"Unknown Calibration Set: {calibration_set_id}")
        set_code = str(target.get("code") or calibration_set_id)
        owner_session_id = str(target.get("owner_session_id") or item.session_id)
        role = infer_calibration_role(item.name)
        participant = participant_id or (infer_participant(item.name) if role == "pupil_calibration" else None)
        destination_root = item.store.calibration_set_dir(str(target["id"])) / "sources" / item.id
        destination = destination_root / item.name
        moved = False
        registered = False
        try:
            with item.lock:
                token.raise_if_cancelled()
                if item.state != 'sealed':
                    raise ValidationError('Upload was cancelled or already finalized')
                item.state = 'processing'
                destination_root.mkdir(parents=True, exist_ok=False)
                item.data.replace(destination)
                moved = True
            metadata = calibration_metadata(destination, set_code)
            result = import_local_asset(
                item.store,
                session_id=owner_session_id,
                path=destination,
                label=label or item.name,
                participant_id=participant,
                checksum=True,
                known_sha256=item.sha.hexdigest(),
                storage_mode='project_copy',
                original_name=item.name,
                metadata_extra=metadata,
                progress=progress,
                cancellation=token,
            )
            asset = result.get('asset') or {}
            updated = set_asset_calibration_scope(
                item.store,
                session_id=owner_session_id,
                asset_id=str(asset.get('id')),
                calibration_set_id=set_code,
                calibration_role=role,
            )
            result['asset'] = updated
            result['calibration_set_id'] = set_code
            result['calibration_role'] = role
            result['owner_session_id'] = owner_session_id
            item.state = 'registered'
            registered = True
            return result
        finally:
            if moved and not registered:
                shutil.rmtree(destination_root, ignore_errors=True)
            shutil.rmtree(item.stage_dir, ignore_errors=True)
            with self._lock:
                self._items.pop(item.id, None)

    def has_pending(self):
        with self._lock:
            return any(i.state in {'receiving','sealed','processing'} for i in self._items.values())

    def cleanup(self):
        with self._lock: items = list(self._items.values())
        for item in items:
            with item.lock:
                if item.state in {'receiving','sealed'}:
                    item.state='cancelled'; shutil.rmtree(item.stage_dir,ignore_errors=True)
                    with self._lock: self._items.pop(item.id,None)
