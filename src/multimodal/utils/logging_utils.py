import csv
import json
import os
from typing import Dict


class MetricLogger:
    def __init__(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        self.jsonl_path = os.path.join(run_dir, "metrics.jsonl")
        self.csv_path = os.path.join(run_dir, "metrics.csv")
        self._csv_header_written = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0

    def log(self, row: Dict) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)

