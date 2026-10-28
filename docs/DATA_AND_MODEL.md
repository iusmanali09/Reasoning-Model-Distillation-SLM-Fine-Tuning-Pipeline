# Data and model governance

## Teacher trace contract

The pipeline accepts explicit `prompt`, `reasoning`, and `answer` strings in JSONL. It does not scrape or request chain-of-thought from a provider. Use only traces you are authorized to process and whose licenses/terms permit derivative training. Minimize personal, confidential, and regulated data before ingestion.

The validator normalizes text, strips control characters, rejects empty or oversized fields, and de-duplicates identical normalized prompts. It does not detect PII, copyright, unsafe content, fabricated reasoning, or semantic duplicates. Human review and domain-specific filters remain necessary.

## Split design and provenance

The default seeded random split is a reproducibility aid, not a leakage guarantee. Split related examples by source, user, task family, or time before running this utility if those groups can correlate. Keep a separate evaluation set that is never used for training or config selection. The preparation manifest contains the source filename and SHA-256, split counts, fraction, seed, and preprocessing summary; store that manifest with the run record but protect it if filenames are sensitive.

## Model selection

Choose a model that supports the tokenizer/chat template used by `apply_chat_template`, fits the target hardware, and has a license compatible with intended use. The sample Qwen model name in `configs/train.yaml` is a configurable example, not a recommendation or verified license grant. Inspect the model card, tokenizer, EOS behavior, architecture, and remote-code requirements before loading it.

QLoRA reduces memory use through quantization and trainable low-rank adapters; it does not guarantee that a 3B/7B model will achieve teacher-level accuracy. Report hardware, base model revision, data manifest, hyperparameters, and task-specific results for every comparison.
