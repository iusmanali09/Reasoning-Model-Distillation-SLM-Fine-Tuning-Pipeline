"""QLoRA SFT training from prepared prompt-completion traces."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    args = parser.parse_args()
    try:
        import torch
        import yaml
        from datasets import load_dataset
        from peft import LoraConfig
        import transformers
        import trl
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(f"Install requirements in a compatible CUDA environment: {exc}") from None

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit("Training config must contain a YAML mapping")
    required = ("model_name_or_path", "data_dir", "output_dir")
    if not all(isinstance(config.get(key), str) and config[key] for key in required):
        raise SystemExit(f"Training config must set: {', '.join(required)}")
    if not torch.cuda.is_available():
        raise SystemExit("This QLoRA example requires an NVIDIA CUDA GPU; no CPU training fallback is configured")
    if bool(config.get("bf16", False)) and not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16=true but the selected CUDA device does not report bfloat16 support")

    data_dir = Path(config["data_dir"])
    train_file, validation_file = data_dir / "train.jsonl", data_dir / "validation.jsonl"
    if not train_file.is_file() or not validation_file.is_file():
        raise SystemExit("Run app.prepare first to create train.jsonl and validation.jsonl")
    datasets = load_dataset("json", data_files={"train": str(train_file), "validation": str(validation_file)})
    model_revision = config.get("model_revision") or None
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        revision=model_revision,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise SystemExit("Tokenizer must define eos_token or pad_token")
        tokenizer.pad_token = tokenizer.eos_token

    quantization = None
    if bool(config.get("load_in_4bit", True)):
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bool(config.get("bf16", False)) else torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        quantization_config=quantization,
        device_map="auto",
        revision=model_revision,
        torch_dtype=torch.bfloat16 if bool(config.get("bf16", False)) else torch.float16,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    model.config.use_cache = False
    lora = LoraConfig(
        r=int(config.get("lora_rank", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    output_dir = Path(config["output_dir"])
    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=int(config.get("max_seq_length", 4096)),
        completion_only_loss=True,
        learning_rate=float(config.get("learning_rate", 1e-4)),
        num_train_epochs=float(config.get("num_train_epochs", 2)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        bf16=bool(config.get("bf16", False)),
        fp16=bool(config.get("fp16", False)),
        gradient_checkpointing=True,
        report_to="none",
        seed=int(config.get("seed", 42)),
    )

    def to_chat_completion(row: dict[str, str]) -> dict[str, list[dict[str, str]]]:
        return {
            "prompt": [
                {"role": "system", "content": "Solve the task carefully. Give a concise rationale and a correct final answer."},
                {"role": "user", "content": row["prompt"]},
            ],
            "completion": [{"role": "assistant", "content": row["completion"]}],
        }

    train_formatted = datasets["train"].map(to_chat_completion, remove_columns=datasets["train"].column_names)
    validation_formatted = datasets["validation"].map(to_chat_completion, remove_columns=datasets["validation"].column_names)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_formatted,
        eval_dataset=validation_formatted,
        processing_class=tokenizer,
        peft_config=lora,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "base_model": config["model_name_or_path"],
        "base_model_revision": model_revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_records": len(datasets["train"]),
        "validation_records": len(datasets["validation"]),
        "training_metrics": train_result.metrics,
        "evaluation_metrics": eval_metrics,
        "seed": int(config.get("seed", 42)),
        "software": {"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__},
        "cuda_device": torch.cuda.get_device_name(0),
        "data_manifest": str(data_dir / "manifest.json") if (data_dir / "manifest.json").is_file() else None,
        "note": "Evaluate on a separate task-family-held-out set before claiming transfer or reasoning improvements.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run-metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Adapter and run metadata saved under {output_dir}")


if __name__ == "__main__":
    main()
