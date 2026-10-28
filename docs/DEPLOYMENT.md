# Deployment and export

1. Pin the exact base-model commit/revision and record the tokenizer revision.
2. Review and approve dataset provenance, license, privacy, deduplication, and split policy.
3. Run preparation and inspect counts, a small reviewed sample, and the generated manifest.
4. Run training manually on approved, isolated GPU infrastructure. Keep logs free of raw prompts and traces.
5. Evaluate on an independent task-family-held-out set, including correctness, refusal, robustness, and regression cases.
6. Record the adapter/base revisions, configuration, hardware, software versions, and evaluation results.
7. Export or merge adapters only after compatibility, license, and safety review. The repository currently does not implement GGUF/AWQ conversion or model publishing.

The repository does not provision a GPU runtime, scheduler, artifact registry, or CI training job. Add those according to your organization's model supply-chain and data controls.
