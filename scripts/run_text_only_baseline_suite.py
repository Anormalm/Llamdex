import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.eval.plan1_eval import EvalPlan1Args, evaluate_plan1
from src.multimodal.trainers.plan1_trainer import TrainPlan1Args, train_plan1


def _load_rows(path: str) -> List[Dict]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if ext == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported file format: {ext}")


def _extract_xy(rows: List[Dict]) -> Tuple[List[str], List[str]]:
    xs = []
    ys = []
    for r in rows:
        x = r.get("description", r.get("text", r.get("report")))
        y = r.get("label", r.get("target", r.get("class")))
        if x is None or y is None:
            continue
        xs.append(str(x))
        ys.append(str(y))
    if not xs:
        raise ValueError("No valid rows; required columns: description/text/report and label/target/class")
    return xs, ys


def _majority_accuracy(train_y: List[str], eval_y: List[str]) -> float:
    from collections import Counter

    major = Counter(train_y).most_common(1)[0][0]
    correct = sum(1 for y in eval_y if y == major)
    return correct / max(1, len(eval_y))


def _tfidf_logreg_accuracy(train_x: List[str], train_y: List[str], eval_x: List[str], eval_y: List[str]) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000)
    xtr = vec.fit_transform(train_x)
    xte = vec.transform(eval_x)
    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    clf.fit(xtr, train_y)
    pred = clf.predict(xte)
    correct = sum(1 for p, g in zip(pred, eval_y) if p == g)
    return correct / max(1, len(eval_y))


def parse_args():
    p = argparse.ArgumentParser(description="Text-only baseline suite for hospital-style privacy setting.")
    p.add_argument("--mistral_models_path", type=str, default="model/llm")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--train_file", type=str, required=True, help="Hospital train file (csv/jsonl).")
    p.add_argument("--eval_file", type=str, required=True, help="Hospital eval file (csv/jsonl).")
    p.add_argument("--run_dir", type=str, default="runs/plan1_hospital_text")
    p.add_argument("--qa_type", type=str, default="yesno", choices=["label", "label_code", "index", "yesno"])
    p.add_argument("--text_encoder_model_id", type=str, default="distilroberta-base")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=512)
    p.add_argument("--num_tokens", type=int, default=8)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--evidence_dim", type=int, default=256)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_csv", type=str, default="runs/hospital_text_baseline_suite.csv")
    p.add_argument("--out_json", type=str, default="runs/hospital_text_baseline_suite.json")
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    os.makedirs(a.run_dir, exist_ok=True)
    if "tiny-random" in a.model_name.lower():
        print("[warning] tiny-random backbone selected; results will be capacity-limited and not industry-representative.")

    # 1) Train injection connectors on hospital text
    train_metrics = train_plan1(
        TrainPlan1Args(
            mistral_models_path=a.mistral_models_path,
            model_name=a.model_name,
            run_dir=a.run_dir,
            dataset_name="hospital_text",
            task_family="single_image",
            evidence_source="text",
            qa_type=a.qa_type,
            hospital_train_file=a.train_file,
            hospital_eval_file=a.eval_file,
            text_encoder_model_id=a.text_encoder_model_id,
            batch_size=a.batch_size,
            num_epochs=a.num_epochs,
            learning_rate=a.learning_rate,
            max_train_samples=a.max_train_samples,
            max_eval_samples=a.max_eval_samples,
            num_tokens=a.num_tokens,
            layer_to_add=a.layer,
            evidence_dim=a.evidence_dim,
            alpha=a.alpha,
            use_adapters=True,
            adapter_bottleneck=64,
            inject_location="post_attn",
            device=a.device,
            seed=a.seed,
        )
    )

    ckpt = os.path.join(a.run_dir, "best_connectors.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(a.run_dir, "last_connectors.pt")

    # 2) Evaluate injection / llm_only
    inj = evaluate_plan1(
        EvalPlan1Args(
            mistral_models_path=a.mistral_models_path,
            model_name=a.model_name,
            connectors_path=ckpt,
            dataset_name="hospital_text",
            hospital_eval_file=a.eval_file,
            task_family="single_image",
            evidence_source="text",
            qa_type=a.qa_type,
            text_encoder_model_id=a.text_encoder_model_id,
            batch_size=a.batch_size,
            max_eval_samples=a.max_eval_samples,
            baseline="injection",
            use_adapters=True,
            adapter_bottleneck=64,
            inject_location="post_attn",
            device=a.device,
        )
    )
    llm = evaluate_plan1(
        EvalPlan1Args(
            mistral_models_path=a.mistral_models_path,
            model_name=a.model_name,
            dataset_name="hospital_text",
            hospital_eval_file=a.eval_file,
            task_family="single_image",
            evidence_source="text",
            qa_type=a.qa_type,
            text_encoder_model_id=a.text_encoder_model_id,
            batch_size=a.batch_size,
            max_eval_samples=a.max_eval_samples,
            baseline="llm_only",
            use_adapters=True,
            adapter_bottleneck=64,
            inject_location="post_attn",
            device=a.device,
        )
    )

    # 3) Industry text baselines
    train_rows = _load_rows(a.train_file)
    eval_rows = _load_rows(a.eval_file)
    train_x, train_y = _extract_xy(train_rows)
    eval_x, eval_y = _extract_xy(eval_rows)
    majority_acc = _majority_accuracy(train_y, eval_y)
    tfidf_acc = _tfidf_logreg_accuracy(train_x, train_y, eval_x, eval_y)

    rows = [
        {
            "model_type": "injection_text_privacy",
            "dataset": "hospital_text",
            "metric_name": "accuracy",
            "metric": float(inj.get("accuracy", 0.0)),
            "notes": "Plan-1 text-first, adapters=on, inject=post_attn",
        },
        {
            "model_type": "llm_only",
            "dataset": "hospital_text",
            "metric_name": "accuracy",
            "metric": float(llm.get("accuracy", 0.0)),
            "notes": "No evidence injection",
        },
        {
            "model_type": "majority_label",
            "dataset": "hospital_text",
            "metric_name": "accuracy",
            "metric": float(majority_acc),
            "notes": "Class-frequency baseline",
        },
        {
            "model_type": "tfidf_logreg",
            "dataset": "hospital_text",
            "metric_name": "accuracy",
            "metric": float(tfidf_acc),
            "notes": "Industry-strong classic text baseline",
        },
        {
            "model_type": "train_status",
            "dataset": "hospital_text",
            "metric_name": "final_eval_acc",
            "metric": float(train_metrics.get("final_eval_acc", 0.0)),
            "notes": "Connector training final eval metric",
        },
    ]

    with open(a.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Completed {len(rows)} rows.")
    print(f"Train metrics: {train_metrics}")
    print(f"Injection eval: {inj}")
    print(f"LLM-only eval: {llm}")
    print(f"CSV: {a.out_csv}")
    print(f"JSON: {a.out_json}")


if __name__ == "__main__":
    main()
