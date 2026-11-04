[English](CONTRIBUTING.md) · [简体中文](docs/i18n/zh-CN/CONTRIBUTING.md) · [日本語](docs/i18n/ja/CONTRIBUTING.md)

# Contributing to RiichiLM

Contributions to models, reinforcement learning, rules, the browser arena, tests, and translations are welcome. Start with the [README](README.md) and [roadmap](ROADMAP.md). Discuss substantial algorithm, protocol, dependency, or architecture changes in an issue before investing in an implementation.

## Issues and pull requests

1. Describe a concrete problem, expected behavior, and a minimal reproducer. Include the exact command, Python/Node and dependency versions, OS/device, and sanitized diagnostics. For rule issues, use a tiny hand-written game state.
2. Keep the change focused. Preserve the boundary between rule validation, data contracts, model learning, and optimizer updates. Tokenization encodes and decodes; it must not grow a second rules engine. Actors must never receive opponents' private observations.
3. Preserve frozen protocol identifiers, tokenizer resources, seed domains, and historical test fixtures. Explain compatibility implications before changing checkpoint or data formats.
4. Run checks appropriate to the change and report commands, results, platform, skipped cases, and remaining limits. Use deterministic synthetic inputs where possible. Timing claims need paired measurements with the same model, precision, context, supervision, and workload.
5. Update affected English, Chinese, and Japanese README, roadmap, contribution text, and UI messages together. Keep technical statements equivalent; do not silently omit prerequisites in a translation.
6. Use an English PR title and description. State the problem, resulting behavior, and validation. Small, reviewable patches are preferred; dependency upgrades belong in a separately explained change.

## Local checks

Use Python 3.14 and one dependency variant from the README. Web development uses Node.js 20.19+ or 22.12+. Examples of scoped checks:

```bash
.venv/bin/python -m pytest tests/unit/tokenizer -q
(cd web && npm ci && npm run build && npm run test:e2e)
```

Browser tests need Chromium available to Playwright; `CHROMIUM_EXECUTABLE` may select an existing installation. Run other relevant model, rules, or RL tests for those changes. A CPU test is not a GPU result, and a short synthetic run is not a playing-strength evaluation. Do not launch a full dataset job or persistent training merely to validate a small patch.

## What to keep out of a contribution

- Training/evaluation datasets, real or private game logs, trained models, Actor snapshots, optimizer states, trajectories, and run outputs. Generate small synthetic examples at test runtime instead.
- API keys, credentials, `.env` files, personal paths, caches, virtual environments, and build output unrelated to an intentional distribution change.
- Private source-specific data processing, internal agent instructions, repository skills, project memory, internal design/review records, and private acceptance archives.
- Features for cheating, live-game assistance, or unauthorized access to online services.

Do not attach private data or weights to public issues, PRs, CI logs, or release artifacts. Redact diagnostics before posting. Dataset and trained-weight contributions are not requested. Project pretrained weights are coming soon; their download links and license will be announced in the README.

## License and attribution

Contribute only code, text, and assets you have the right to share under the project's [Apache-2.0 license](LICENSE). Preserve third-party licenses, source notices, and [NOTICE](NOTICE); identify new third-party material and AI-generated visual assets. Do not assume code licensing grants rights to game records or model weights. The learning/research statement is project intent and does not replace Apache-2.0.
