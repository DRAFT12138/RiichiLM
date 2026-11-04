[English](ROADMAP.md) · [简体中文](docs/i18n/zh-CN/ROADMAP.md) · [日本語](docs/i18n/ja/ROADMAP.md)

# Roadmap

RiichiLM's current capabilities and next priorities. Priorities may change; playing strength and long-run training quality require separate evaluation.

## Available

- [x] Mahjong tokenizer, legal-action trie, Qwen3 policies, and Rust rules engine.
- [x] Action-token supervision, shared-context training, and full checkpoint recovery.
- [x] Self-play collection, Oracle Critic, KL-anchored PPO, frozen-opponent pool, and RL round-boundary recovery.
- [x] Human-vs-AI play, four-AI spectating, and headless matches with local NTP checkpoints, custom weights, external LLMs, or random players.
- [x] English, Chinese, and Japanese interfaces and documentation; public repository and CI.

These items describe implementation. Recorded CPU/browser and AMD GPU checks have their own scope; independent CUDA runtime validation remains open.

## Coming soon: pretrained weights

**Pretrained weights are coming soon as an open-source release.**

- [ ] Publish pretrained weights with download links, the weights license, model configuration, and loading instructions.
- [ ] Verify the published artifact with the documented Web and headless inference commands.

The [README](README.md) will announce availability. Training and evaluation datasets remain outside the release. Inference weights alone do not provide the optimizer, RNG, data cursor, and compatible corpus required for full training recovery.

## Experimentation and evaluation

- [ ] Add training-health metrics for gradients, activations, actual parameter updates, KL, entropy, and Critic/advantage behavior; measure overhead.
- [ ] Evaluate long-run shared-context throughput, numerical behavior, and training quality with reproducible inputs.
- [ ] Broaden independent CUDA installation/runtime validation and native-build coverage.
- [ ] Measure Web model latency and memory across workloads; investigate RL Actor support in the arena.
- [ ] Improve three-language terminology, accessibility, and narrow-screen layouts.
- [ ] Run controlled Oracle/PPO/KL ablations, sustained self-play, and frozen-pool promotion/turnover evaluations.
- [ ] Report seat-rotated comparisons, uncertainty, compute cost, and reproducible evaluation procedures.

## Distribution

For learning and research. Training datasets, evaluation datasets, and private game records are not distributed. The upcoming weights release will specify its own license; the code remains Apache-2.0. Use compatible, appropriately licensed inputs for training.

[Get started](README.md) · [Contribute](CONTRIBUTING.md)
