# Development guide

## Components

- `app/prepare.py`: validates and normalizes teacher trace JSONL, deduplicates exact prompts, and creates deterministic splits and a manifest.
- `app/train.py`: performs an explicitly invoked TRL supervised fine-tuning run with PEFT and optional 4-bit quantization.
- `configs/train.yaml`: model, data, LoRA, and training settings.
- `data/teacher.example.jsonl`: a synthetic example only.

Keep external model loading behind explicit configuration. Do not enable `trust_remote_code` by default, add automatic dataset downloads, or write artifacts into source-controlled paths. Follow the shared [Usman engineering standards](../../CODE_STANDARDS.md).

## Training implementation

The script expects prompt/completion data, applies the chosen tokenizer's chat template, uses `SFTConfig` and `SFTTrainer`, and requests PEFT LoRA. When `load_in_4bit` is true, it creates an NF4 `BitsAndBytesConfig`. The script fails early without CUDA and checks bfloat16 support when configured. For non-NVIDIA systems, choose an explicit supported backend and implement/validate it before changing the CUDA guard.

Do not train during CI. Keep evaluation data and checkpoints outside version control. Validate adapter loading and inference using the exact base model revision before export.

## API references

- [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/main/sft_trainer)
- [TRL Unsloth integration](https://huggingface.co/docs/trl/main/unsloth_integration)
