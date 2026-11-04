from __future__ import annotations

import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock
from typing import Any, Callable

from oculostudio.domain.models import JobRecord, JobStatus
from oculostudio.services.idempotency import (
    SIGNATURE_VERSION,
    find_artifact_by_signature,
    find_job_by_signature,
    iter_result_artifact_ids,
    processing_input_context,
    processing_signature,
    reused_job_for_artifact,
)
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, utc_now


class JobManager:
    def __init__(self, store_provider: Callable[[], ProjectStore], *, max_workers: int = 2):
        self.store_provider = store_provider
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="oculostudio")
        self._futures: dict[str, Future] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._stores: dict[str, ProjectStore] = {}
        self._lock = RLock()

    def submit_idempotent(
        self,
        *,
        operation: str,
        label: str,
        session_id: str | None,
        parameters: dict[str, Any],
        function: Callable[..., dict[str, Any]],
        kwargs: dict[str, Any],
        force_recompute: bool = False,
    ) -> tuple[JobRecord, dict[str, Any]]:
        """Submit one immutable processing job, or reuse an identical result.

        The decision is made while holding both the manager and project locks,
        so rapid double-clicks and concurrent requests cannot enqueue the same
        scientific operation twice. Completed results remain reusable after
        the visible job history has been cleared because the signature is also
        written into artifact provenance.
        """
        store = self.store_provider()
        snapshot = store.snapshot()
        signature = processing_signature(
            operation,
            session_id,
            parameters,
            input_context=processing_input_context(snapshot, session_id, parameters),
        )
        decision: dict[str, Any] = {}
        new_job = JobRecord(
            operation=operation,
            label=label,
            session_id=session_id,
            parameters={
                **parameters,
                "_processing_signature": signature,
                "_signature_version": SIGNATURE_VERSION,
            },
        )

        with self._lock:
            def decide(project):
                if not force_recompute:
                    existing_job = find_job_by_signature(project, signature)
                    if existing_job is not None:
                        decision.update(
                            job=existing_job,
                            reused=True,
                            reason=(
                                "active_job"
                                if existing_job.status in {JobStatus.QUEUED, JobStatus.RUNNING}
                                else "completed_result"
                            ),
                        )
                        return
                    artifact = find_artifact_by_signature(project, session_id, signature)
                    if artifact is not None:
                        reuse_job = reused_job_for_artifact(
                            operation=operation,
                            label=label,
                            session_id=session_id,
                            parameters=parameters,
                            signature=signature,
                            artifact=artifact,
                        )
                        project.jobs[reuse_job.id] = reuse_job
                        decision.update(
                            job=reuse_job,
                            reused=True,
                            reason="completed_artifact",
                            artifact_id=artifact.id,
                        )
                        return
                project.jobs[new_job.id] = new_job
                decision.update(job=new_job, reused=False, reason="submitted")

            store.update(decide)
            job = decision["job"]
            if decision["reused"]:
                return job, {
                    "reused": True,
                    "reason": decision["reason"],
                    "processing_signature": signature,
                    "artifact_id": decision.get("artifact_id")
                    or job.result.get("artifact", {}).get("id"),
                    "message": (
                        "An identical job is already queued or running."
                        if decision["reason"] == "active_job"
                        else "An identical immutable result already exists and was reused."
                    ),
                }

            token = CancellationToken.create()
            self._tokens[job.id] = token
            self._stores[job.id] = store
            self._futures[job.id] = self.executor.submit(
                self._run_job, store, job.id, function, kwargs, token
            )
        return job, {
            "reused": False,
            "reason": "submitted",
            "processing_signature": signature,
            "artifact_id": None,
            "message": "A new immutable processing job was submitted.",
        }

    def submit(
        self,
        *,
        operation: str,
        label: str,
        session_id: str | None,
        parameters: dict[str, Any],
        function: Callable[..., dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> JobRecord:
        store = self.store_provider()
        job = JobRecord(
            operation=operation,
            label=label,
            session_id=session_id,
            parameters=parameters,
        )
        token = CancellationToken.create()

        def add(project):
            project.jobs[job.id] = job

        store.update(add)
        with self._lock:
            self._tokens[job.id] = token
            self._stores[job.id] = store
            self._futures[job.id] = self.executor.submit(
                self._run_job, store, job.id, function, kwargs, token
            )
        return job

    def _run_job(
        self,
        store: ProjectStore,
        job_id: str,
        function: Callable[..., dict[str, Any]],
        kwargs: dict[str, Any],
        token: CancellationToken,
    ) -> None:
        def start(project):
            job = project.jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = utc_now()
            job.stage = "Starting"
            job.progress = 0.0

        store.update(start)

        def progress(fraction: float, stage: str) -> None:
            def mutate(project):
                job = project.jobs[job_id]
                if job.status == JobStatus.CANCELLED:
                    return
                job.progress = fraction
                job.stage = stage
                if not job.log or job.log[-1] != stage:
                    job.log.append(stage)
                    if len(job.log) > 300:
                        del job.log[:-300]

            store.update(mutate)

        try:
            result = function(**kwargs, progress=progress, cancellation=token)
            if token.cancelled:
                raise RuntimeError("cancelled")

            def complete(project):
                job = project.jobs[job_id]
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                job.stage = "Completed"
                job.completed_at = utc_now()
                job.result = result
                signature = job.parameters.get("_processing_signature")
                if signature and job.session_id in project.sessions:
                    session = project.sessions[job.session_id]
                    for artifact_id in dict.fromkeys(iter_result_artifact_ids(result)):
                        artifact = session.artifacts.get(artifact_id)
                        if artifact is None:
                            continue
                        artifact.provenance["processing_signature"] = signature
                        artifact.provenance["processing_operation"] = job.operation
                        artifact.provenance["signature_version"] = SIGNATURE_VERSION

            store.update(complete)
        except Exception as exc:
            cancelled = token.cancelled or str(exc).lower() == "cancelled" or exc.__class__.__name__ == "CancelledError"

            def fail(project):
                job = project.jobs[job_id]
                job.status = JobStatus.CANCELLED if cancelled else JobStatus.FAILED
                job.stage = "Cancelled" if cancelled else "Failed"
                job.completed_at = utc_now()
                job.error = str(exc)
                if not cancelled:
                    job.log.append(traceback.format_exc())

            store.update(fail)
        finally:
            with self._lock:
                self._tokens.pop(job_id, None)
                self._stores.pop(job_id, None)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            token = self._tokens.get(job_id)
            future = self._futures.get(job_id)
        if token is None:
            return False
        token.cancel()
        if future:
            future.cancel()
        store = self._stores.get(job_id) or self.store_provider()

        def mutate(project):
            if job_id in project.jobs and project.jobs[job_id].status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                project.jobs[job_id].status = JobStatus.CANCELLED
                project.jobs[job_id].stage = "Cancelling"

        store.update(mutate)
        return True

    def clear_finished(self) -> int:
        store = self.store_provider()
        removed = 0

        def mutate(project):
            nonlocal removed
            for job_id, job in list(project.jobs.items()):
                if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
                    del project.jobs[job_id]
                    removed += 1

        store.update(mutate)
        return removed

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
