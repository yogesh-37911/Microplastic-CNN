"""Train and compare supported image classifiers on a validated dataset."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ml.evaluation import evaluate_model
from app.ml.training import train_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Folder with microplastic/ and non_microplastic/ subfolders")
    parser.add_argument("--architectures", nargs="+", default=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the pretrained classifier head for a faster CPU run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/reports/model_comparison.json")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    report: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "methodology": {
            "split": "source-grouped stratified 70/15/15",
            "model_selection": "validation macro-F1",
            "final_evaluation": "held-out test set, once per selected checkpoint",
            "augmentation": "training split only",
            "freeze_backbone": args.freeze_backbone,
            "image_size": args.image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "runs": [],
    }

    from app.ml.training import discover_dataset
    _, _, class_names, class_counts = discover_dataset(str(dataset))
    if set(class_names) != {"microplastic", "non_microplastic"}:
        raise SystemExit("Dataset must have exactly microplastic/ and non_microplastic/ folders.")
    if min(class_counts.values()) < 50:
        raise SystemExit(f"Training requires at least 50 images per class; found {class_counts}.")
    try:
        from app.ml.training import split_dataset
        split_dataset(*discover_dataset(str(dataset))[:2], seed=args.seed)
    except ValueError as exc:
        raise SystemExit(f"Dataset not ready for group-safe evaluation: {exc}") from exc

    for architecture in args.architectures:
        print(f"\n=== {architecture} ===", flush=True)
        result = train_model(
            dataset_path=str(dataset),
            architecture=architecture,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
            seed=args.seed,
            use_class_weights=True,
            freeze_backbone=args.freeze_backbone,
        )
        metrics = evaluate_model(
            checkpoint_path=result["best_checkpoint_path"],
            architecture=architecture,
            test_paths=result["test_paths"],
            test_labels=result["test_labels"],
            class_names=result["class_names"],
            image_size=args.image_size,
        )
        checkpoint = torch.load(
            result["best_checkpoint_path"], map_location="cpu", weights_only=False
        )
        checkpoint["test_metrics"] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["f1_macro"],
            "macro_precision": metrics["precision_macro"],
            "macro_recall": metrics["recall_macro"],
            "confusion_matrix": metrics["confusion_matrix"],
            "count": metrics["test_count"],
        }
        checkpoint["split_method"] = result["split_method"]
        checkpoint["split_groups"] = result["split_groups"]
        torch.save(checkpoint, result["best_checkpoint_path"])
        report["runs"].append({
            "architecture": architecture,
            "class_counts": result["class_counts"],
            "train_count": result["train_count"],
            "validation_count": result["val_count"],
            "test_count": result["test_count"],
            "best_validation_macro_f1": max((h["val_macro_f1"] for h in result["training_history"]), default=None),
            "training_history": result["training_history"],
            "held_out_test": metrics,
            "checkpoint": result["best_checkpoint_path"],
            "split_groups": result["split_groups"],
        })

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nComparison report saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
