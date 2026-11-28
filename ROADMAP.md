[English](ROADMAP.md) · [简体中文](docs/i18n/zh-CN/ROADMAP.md) · [日本語](docs/i18n/ja/ROADMAP.md)

# Roadmap

RiichiLM's current capabilities and next priorities. Priorities may change; playing strength and long-run training quality require separate evaluation.

## Available

- [x] Mahjong tokenizer, legal-action trie, Qwen3 policies, and Rust rules engine.
- [x] Action-token supervision, shared-context training, and full checkpoint recovery.
- [x] Self-play collection, Oracle Critic, KL-anchored PPO, frozen-opponent pool, and RL round-boundary recovery.
- [x] Human-vs-AI play, four-AI spectating, and headless matches with local NTP checkpoints, custom weights, external LLMs, or random players, including configurable `temperature` and `top-p` inference sampling.
- [x] English, Chinese, and Japanese interfaces and documentation; public repository and CI.

These items describe implementation. Recorded CPU/browser and AMD GPU checks have their own scope; independent CUDA runtime validation remains open.

## Pretrained base weights

**The 500K supervised pretrained base weights are published on Hugging Face at [`Draft12138/RiichiLM-Base`](https://huggingface.co/Draft12138/RiichiLM-Base)** (`riichilm-base-500k.pt`, **CC-BY-NC-4.0**), while pretraining continues toward the **1M-step** base checkpoint (`riichilm-base-1m.pt`) planned for release in the same repository. Open-weight releases are strictly limited to these two supervised pretrained base checkpoints (`500K` and `1M`); subsequent RL-stage artifacts (RL policy weights and related auxiliary models) will not be released as open weights, and future RL roadmap/design plans are not publicly disclosed at this time.

- [x] Publish the 500K pretrained base model weights (`riichilm-base-500k.pt`), SHA-256 checksums, `CC-BY-NC-4.0` license, model configuration, and loading instructions at [`Draft12138/RiichiLM-Base`](https://huggingface.co/Draft12138/RiichiLM-Base).
- [x] Verify the downloaded 500K Hugging Face artifact via SHA-256 checksum, `weights_only=True` loading, Web catalog resolution, and a single-round GPU (`cuda:0`) headless `arena-evaluate` run.
- [ ] Complete **1M-step** pretraining, publish `riichilm-base-1m.pt` and the 1M model card to `Draft12138/RiichiLM-Base`, and verify the published artifact.

See the [README](README.md) and [500K model card](docs/model-card-500k.md) for usage details. Training and evaluation datasets remain outside the release. Inference weights alone do not provide the optimizer, RNG, data cursor, and compatible corpus required for full training recovery.

## Experimentation and evaluation

- [ ] Add training-health metrics for gradients, activations, and actual parameter updates; measure overhead.
- [ ] Evaluate long-run shared-context throughput, numerical behavior, and training quality with reproducible inputs.
- [ ] Broaden independent CUDA installation/runtime validation and native-build coverage.
- [ ] Measure Web model latency and memory across workloads.
- [ ] Add standalone Nanikiru evaluation for exported weights and multi-checkpoint Elo/style benchmarking.
- [ ] Improve three-language terminology, accessibility, and narrow-screen layouts.
- [ ] Report statistically powered four-seat-rotated comparisons (with uncertainty intervals, style diagnostics, and compute cost) and reproducible evaluation procedures.

## Distribution

For learning and research. Training datasets, evaluation datasets, and private game records are not distributed. Open-weight releases are limited to the `500K` and planned `1M` supervised pretrained base models under **CC-BY-NC-4.0** and exclude subsequent RL-stage artifacts (RL policy weights and related auxiliary models); the codebase remains **Apache-2.0**. Use compatible, appropriately licensed inputs for training.

[Get started](README.md) · [Contribute](CONTRIBUTING.md)
