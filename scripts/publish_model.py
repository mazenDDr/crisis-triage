"""Publish the fine-tuned Laya (data/laya_ft) to the Hugging Face Hub, then check the upload.

Builds the model card from the committed results, creates the public model repo, uploads the
folder, then loads the model back from the Hub into an empty cache and compares its answers
with the local answers on 20 dev-sample messages.

  python scripts/publish_model.py   (gpu-box, logged in to the Hub)
"""

import importlib.util
import os
import tempfile
from pathlib import Path

REPO_ID = "mazenDDr/laya-crisis-triage"
FOLDER = Path("data/laya_ft")


def build_card():
    spec = importlib.util.spec_from_file_location("card", "scripts/build_model_card.py")
    card = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(card)
    card.main(FOLDER / "README.md")


def upload() -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)
    info = api.upload_folder(
        repo_id=REPO_ID, folder_path=str(FOLDER), commit_message="Upload laya-crisis-triage"
    )
    return info.commit_url


def verify(n: int = 20) -> float:
    """Largest difference between Hub and local answers on n HumAID dev-sample messages."""
    import laya

    from crisis_triage.laya_run import load
    from crisis_triage.runs import run_questions
    from crisis_triage.scoring import dev_sample

    local = load(Path("outputs/t07/humaid__dev.jsonl"))
    qs = run_questions("humaid", "coarse_described")
    with tempfile.TemporaryDirectory() as cache:
        os.environ["HF_HUB_CACHE"] = cache
        agent = laya.load(REPO_ID, device="cuda")
        worst = 0.0
        for r in dev_sample("humaid").head(n).to_dict("records"):
            res = agent.predict({"message": r["text"]}, qs)["answers"]
            probs = res["coarse_described/label"]["probabilities"]
            saved = local[r["uid"]]["answers"]["coarse_described/label"]["probs"]
            worst = max(worst, max(abs(probs[k] - saved[k]) for k in saved))
    return worst


if __name__ == "__main__":
    build_card()
    print("uploaded:", upload(), flush=True)
    print(f"largest Hub vs local difference: {verify():.5f}", flush=True)
