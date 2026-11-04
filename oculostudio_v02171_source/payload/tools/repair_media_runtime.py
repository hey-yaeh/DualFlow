"""In-place media repair for OculoStudio 0.5.3+; does not run pip or touch a project."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')


def main():
    from oculostudio.media_runtime import install_windows_runtime
    from oculostudio.errors import OculoStudioError
    print('Checking/installing BOTH FFmpeg and ffprobe in this OculoStudio installation.', flush=True)
    print('Bundled local tools first. Only missing tools require an HTTPS download.', flush=True)
    try:
        result = install_windows_runtime(progress=lambda f,s:print(f'[{f:6.1%}] {s}', flush=True))
        print(json.dumps(result, ensure_ascii=False, indent=2),flush=True)
        print('PASS: encoding AND actual video frame timestamp reading passed.',flush=True)
        print('Reopen Studio, keep your original start/duration, then Preview again.',flush=True)
        return 0
    except (OculoStudioError, OSError) as exc:
        print('REPAIR FAILED: '+str(exc),flush=True)
        print('Original files/projects were not changed. Review the FIRST error above.',flush=True)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
