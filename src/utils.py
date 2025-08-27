import os
import datetime
import shutil
import tempfile
import threading
import time
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, create_repo, whoami
# ----------------------- Periodic HF pusher ------------------------

class HFPusher:
    """
    Periodically uploads a local file to a Hugging Face dataset repository.
    Each push copies the local file to a temp file first for atomicity.
    """
    def __init__(self, repo_id, local_path, path_in_repo, interval_sec=1800, private=True):
        """
        repo_id: e.g. 'your-username/my-new-dataset'
        local_path: local jsonl to upload
        path_in_repo: path inside repo, e.g. 'data/new_dataset.jsonl'
        interval_sec: push frequency (default 30 min)
        """
        self.repo_id = repo_id
        self.local_path = local_path
        self.path_in_repo = path_in_repo
        self.interval_sec = max(60, int(interval_sec))
        self._stop = threading.Event()
        self._thread = None

        self.api = HfApi()
        # Ensure repo exists
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    def _upload_once(self, message_prefix="autosave"):
        if not os.path.exists(self.local_path):
            return
        # Copy to temp to avoid uploading a file that's being written to
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            shutil.copy2(self.local_path, tmp_path)
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            commit_msg = f"{message_prefix}: {os.path.basename(self.local_path)} @ {ts}"
            self.api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=self.path_in_repo,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=commit_msg
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        # Push immediately once so the file appears early (if present)
        try:
            self._upload_once(message_prefix="initial autosave")
        except Exception as e:
            print(f"[HFPusher] Initial upload failed: {e}")

        while not self._stop.wait(self.interval_sec):
            try:
                self._upload_once()
                print("[HFPusher] Autosaved to HF.")
            except Exception as e:
                print(f"[HFPusher] Autosave failed: {e}")

    def stop_and_final_push(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 5)
        try:
            self._upload_once(message_prefix="final autosave")
            print("[HFPusher] Final push complete.")
        except Exception as e:
            print(f"[HFPusher] Final push failed: {e}")