import os
from datetime import datetime, timezone
import shutil
import tempfile
import threading
import time
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, create_repo, whoami
# ----------------------- Periodic HF pusher ------------------------

class HFPusher:
    def __init__(self, repo_id, local_path, path_in_repo, interval_sec=1800, private=True):
        self.repo_id = repo_id
        self.local_path = local_path
        self.path_in_repo = path_in_repo
        self.interval_sec = max(60, int(interval_sec))
        self._stop = threading.Event()
        self._thread = None

        self.api = HfApi()
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    def _upload_once(self, message_prefix="autosave"):
        if not os.path.exists(self.local_path):
            print(f"[HFPusher] Local file not found yet: {self.local_path}")
            return

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            shutil.copy2(self.local_path, tmp_path)
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            commit_msg = f"{message_prefix}: {os.path.basename(self.local_path)} @ {ts}"
            print(f"[HFPusher] Uploading {self.local_path} -> {self.repo_id}:{self.path_in_repo}")
            self.api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=self.path_in_repo,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=commit_msg
            )
            print(f"[HFPusher] Uploaded. Commit message: {commit_msg}")
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