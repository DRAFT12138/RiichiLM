[English](ROADMAP.md) · [简体中文](docs/i18n/zh-CN/ROADMAP.md) · [日本語](docs/i18n/ja/ROADMAP.md)

# Roadmap

This roadmap starts a new public-facing baseline for RiichiLM. It describes capabilities and verifiable next steps, rather than private experiments or historical training schedules. Priorities can change; there are no promised release dates or playing-strength claims.

## Available in the codebase

- [x] Mahjong tokenizer, legal-action trie, Qwen3-based policies, and Rust-backed rules.
- [x] Action-token supervised training, shared-context training, and checkpoint recovery.
- [x] Self-play collection, Oracle Critic, KL-anchored PPO, frozen-opponent pool, and RL round-boundary recovery.
- [x] Local human-vs-AI play and four-AI spectating with independently selected NTP, external LLM, and random players.
- [x] English, Chinese, and Japanese interfaces and project documentation.

These items describe implementation. Competitive strength and long-run quality remain separate evaluation tasks.

## 1. A reproducible public release

- [x] Finalize an explicit file allowlist covering the model, training interfaces, RL, generic runtime/data components, Web, selected tests, and required resources.
- [x] Make public CLI/configuration/dependency references self-contained. Make the current Nanikiru evaluation input independently configurable so an unbundled private evaluation set is not an implicit requirement.
- [x] Document the user-supplied corpus contract and checkpoint prerequisites in the three README translations. Keep source-specific parsers and processing private.
- [x] Run installation and meaningful regression from the exported tree, using the existing dependency environment without importing private source. Cover model construction, random Actor round trips, full game collection, PPO updates, saving, and recovery in a new process.
- [x] Create the separate public repository with independent Git history and public CI.
- [x] Provide a private-side, manually invoked synchronization script with preview, explicit file management, conflict handling, and repeatable exports. No automatic synchronization or force push.

Hosted CI execution and CUDA runtime validation remain separate from the completed local CPU/browser checks.

## 2. Better experimentation

- [ ] Add training-health observability for gradients, activations, actual parameter updates, KL, entropy, and Critic/advantage behavior; measure its overhead.
- [ ] Evaluate long-run shared-context throughput and numerical behavior using contributor-provided inputs and reproducible synthetic cases.
- [ ] Broaden independent CUDA installation/runtime validation and native-build coverage; keep platform-specific evidence explicit.
- [ ] Measure real-model Web latency and memory; investigate RL Actor support in the arena without weakening loading or observation boundaries.
- [ ] Improve three-language terminology, accessibility, and narrow-screen layouts as the interface evolves.

## 3. Research and evaluation

- [ ] Run controlled Oracle/PPO/KL ablations and longer self-play experiments.
- [ ] Evaluate frozen-pool turnover, candidate promotion, and quality gates over sustained runs.
- [ ] Report seat-rotated comparisons, uncertainty, compute cost, and reproducible evaluation procedures. Do not infer strength from a smoke test or a loss curve.

## Distribution boundary

Learning and research only. We will not distribute training datasets, evaluation datasets, pretrained models, trained checkpoints, or private game records. Use your own compatible inputs. Public documentation consists of the README, ROADMAP, and CONTRIBUTING families in English, Chinese, and Japanese, plus necessary license/attribution material and referenced visual assets. Repository skills, AGENTS.md, internal designs, project memory, acceptance evidence, archives, private processing code, secrets, and private Git history stay out of the public export.

[Get started](README.md) · [Contribute](CONTRIBUTING.md)
