from utils import HFPusher
from time import sleep
out = "teacher0_dataset.jsonl"
with open(out, "w", encoding="utf-8") as f:
    f.write('{"hello": "world"}\n')

p = HFPusher(
    repo_id="MinTR-KIEU/teacher0-dataset",
    local_path=out,
    path_in_repo="data/teacher0_dataset.jsonl",
    interval_sec=60,  # 1 minute
    private=True
)
p.start()
sleep(65)  # wait for one autosave
p.stop_and_final_push()
