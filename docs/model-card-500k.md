# RiichiLM 500K Model Card

## Overview

| Field | Details |
| --- | --- |
| Model | RiichiLM legal-action policy model |
| Training stage | Supervised next-action training from randomly initialized weights |
| Optimizer updates | 500,000 |
| Parameters | 101,477,120 (about 101.5 million) |
| Architecture | Qwen3-style decoder, 16 layers, hidden size 768, 12 attention heads, 4 key/value heads |
| Vocabulary | 1,024 tokens |
| Maximum context | 8,192 tokens |
| Task | Predict the next legal Mahjong action from the visible game history and legal-action set |

This checkpoint is a task-specific Riichi Mahjong policy model. It is not a general-purpose chat model.

## Release & Download

- **Hugging Face repository**: [`Draft12138/RiichiLM-Base`](https://huggingface.co/Draft12138/RiichiLM-Base) (released 2026-11-28)
- **Weight file**: `riichilm-base-500k.pt` (FP32, 409,112,985 bytes / ~390 MiB, with sidecar `riichilm-base-500k.json` and `SHA256SUMS.txt`)
- **SHA-256**: `1ed015d480b90790cd843c871f8162bbaae3019a408c4a082432b7e5e24a9b86`
- **Weights license**: **CC-BY-NC-4.0** (codebase remains Apache-2.0)
- **Open-weight scope**: Open-weight releases are limited to this `500K` base checkpoint and the planned `1M` base checkpoint (`riichilm-base-1m.pt`). Subsequent RL-stage artifacts (RL policy weights and related auxiliary models) are not released as open weights, while RL training and evaluation code remains open source.

## Training volume

The corpus snapshot recorded on 2026-10-24 contained 4,585,517 games, 2,439,932,818 supervised actions, and 9,024,927,373 supervised target tokens.

By the end of the 500,000-update run, the training progress record counted **151,877,118 examples updated**. The 500,000 updates processed 24,000,000 batches, with gradient accumulation set to 8.

The model is trained to predict supervised action targets. The legal-action set is supplied separately from the token sequence and constrains the policy output to actions valid in the current state.

## Compute and software

- Accelerators: 6 × NVIDIA GeForce RTX 3090 (24 GB each)
- Parallelism: Distributed Data Parallel (DDP), 6 ranks; no NVLink was available in the recorded host setup
- Precision: BF16 mixed precision; FP32 parameters, gradients, and Adam states
- Accumulation: 8 microbatches per optimizer update
- Recorded training time: approximately 318.1 accelerator-active wall-clock hours across the six training attempts, or approximately 1,909 GPU-hours (six GPUs multiplied by summed training-phase time). This excludes initialization, validation, and checkpointing phases as tracked by the run observer.
- Recorded DDP software environment: NVIDIA driver 515.105.01, CUDA 11.7 toolchain, PyTorch 2.5.1, BF16 support. The 500,000-update run used the six-GPU DDP setup described above.

The run reached 500,000 optimizer updates on 2026-11-16. It began from random initialization and continued through saved training states; it did not start from a pretrained Qwen checkpoint.

## Intended use

Use this model for Riichi Mahjong policy inference in the project’s compatible tokenizer, rule engine, and legal-action interface. It can select among legal actions based on the current player’s observation and prior visible game history.

## Limitations

The model’s behavior reflects the coverage and biases of the training records and preprocessing. Its outputs should be consumed with the matching legal-action constraints and game-state implementation. Standalone inference weights (`riichilm-base-500k.pt`) are distributed on Hugging Face under CC-BY-NC-4.0; training data, evaluation datasets, optimizer states, RNG states, and data cursors are not distributed.

## Record dates

The run reached 500,000 updates on 2026-11-16. Corpus-size figures reflect a snapshot dated 2026-10-24.
