"""Targeted static validation for OculoStudio v0.21.71 Body AOI result handoff."""
from __future__ import annotations

import json
from pathlib import Path

import oculostudio


def main() -> int:
    static_root = Path(oculostudio.__file__).resolve().parent / "static"
    app = (static_root / "app.js").read_text(encoding="utf-8")
    css = (static_root / "styles.css").read_text(encoding="utf-8")
    index = (static_root / "index.html").read_text(encoding="utf-8")

    checks = {
        "installed_version": oculostudio.__version__ == "0.21.71",
        "cache_busting_version": "app.js?v=0.21.71" in index and "styles.css?v=0.21.71" in index,
        "body_job_is_tracked": "pendingBodyAOIJobId" in app and "appState.pendingBodyAOIJobId=queued?.job?.id||null" in app,
        "completion_is_observed": "completedBodyAOIJob" in app and "jobResultArtifactId(completedBodyAOIJob)" in app,
        "result_auto_selects_body_review": all(
            token in app
            for token in (
                "appState.aoiReviewInitialMode='body'",
                "appState.aoiReviewInitialArtifactId=artifactId",
                "fullRender=true",
            )
        ),
        "review_player_receives_initial_artifact": all(
            token in app
            for token in (
                "initialMode:appState.aoiReviewInitialMode",
                "initialArtifactId:appState.aoiReviewInitialArtifactId",
                "OculoReview.bind({sessionId:selectedSessionId(),initialMode,initialArtifactId,initialVideoArtifactId})",
            )
        ),
        "persisted_completed_jobs_have_open_action": "Open Body Human Review" in app and "openAOIReviewResult" in app,
        "successful_job_wording_is_unambiguous": "completed:'ready'" in app and "Result ready" in app,
        "completed_card_has_success_style": ".job-card.status-completed" in css and ".job-result" in css,
        "clear_button_is_textual": ">Clear</button>" in index and "⌫" not in index,
    }
    failed = [name for name, passed in checks.items() if not passed]
    print(json.dumps({"release": "OculoStudio v0.21.71", "checks": checks, "failed": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
