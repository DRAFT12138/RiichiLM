[English](../../../ROADMAP.md) · [简体中文](ROADMAP.md) · [日本語](../ja/ROADMAP.md)

# 路线图

RiichiLM 的现有能力与后续优先事项。优先级可调整；棋力和长期训练质量需要单独评估。

## 已有能力

- [x] 麻将 Tokenizer、合法动作 trie、Qwen3 策略及 Rust 规则引擎。
- [x] 动作 token 监督、共享上下文训练及完整 checkpoint 恢复。
- [x] 自我对弈采集、Oracle Critic、KL 约束 PPO、冻结对手池及 RL 轮边界恢复。
- [x] 真人对战、四 AI 观战及无界面对战，支持本地 NTP checkpoint、自定义权重、外接 LLM 与随机策略，并支持 `temperature` 与 `top-p` 推理采样控制。
- [x] 中英日界面与文档、公开仓库及 CI。

上述勾选表示代码能力。已有 CPU／浏览器和 AMD GPU 验证各有范围；CUDA 独立运行验证仍待完成。

## 预训练基座权重

**50 万步监督预训练基座权重已发布于 Hugging Face：[`Draft12138/RiichiLM-Base`](https://huggingface.co/Draft12138/RiichiLM-Base)**（`riichilm-base-500k.pt`，**CC-BY-NC-4.0**），同时正继续预训练至 **100 万步（1M）** 目标底座（计划发布 `riichilm-base-1m.pt`）。权重开源仅限这两个监督预训练基座模型（`500K` 与 `1M`）；后续强化学习（RL）阶段产物（含 RL 策略权重及相关辅助模型）不在开源权重发布范围内，后续 RL 路线与方案暂不公开。

- [x] 在 [`Draft12138/RiichiLM-Base`](https://huggingface.co/Draft12138/RiichiLM-Base) 发布 50 万步预训练基座权重（`riichilm-base-500k.pt`）、SHA-256 校验、CC-BY-NC-4.0 许可、模型配置与加载说明。
- [x] 回拉验证已发布的 50 万步 Hugging Face 产物：SHA-256 校验、`weights_only=True` 加载、Web 模型目录解析及单局 GPU（`cuda:0`）无界面 `arena-evaluate` 实测通过。
- [ ] 完成 **100 万步（1M）** 预训练，在同一仓库发布 `riichilm-base-1m.pt` 及 1M 模型卡并验证公开产物。

下载与使用说明见 [README](README.md) 与 [50 万步模型卡](model-card-500k.md)。训练和评测数据集仍不分发。推理权重本身不包含完整续训所需的优化器、RNG、数据游标及兼容语料。

## 实验与评估

- [ ] 接入梯度、激活及实际参数更新幅等训练健康指标，并测量开销。
- [ ] 使用可复现输入评估共享上下文的长期吞吐、数值行为与训练质量。
- [ ] 扩大 CUDA 独立安装／运行及原生构建验证。
- [ ] 测量不同负载下的 Web 模型延迟和显存。
- [ ] 增加面向导出权重的独立何切（Nanikiru）评测与多检查点 Elo／风格对比。
- [ ] 改进三语术语、无障碍和窄屏布局。
- [ ] 报告具备统计效力的四席换座对比（含置信区间、对局风格指标与计算成本）及可复现评估流程。

## 分发说明

用于学习与研究，不分发训练数据集、评测数据集或私有牌谱。权重开源仅限 `500K` 与计划中的 `1M` 监督预训练基座模型（采用 **CC-BY-NC-4.0**），不包含后续 RL 阶段产物（含 RL 策略权重及相关辅助模型）；代码保持 **Apache-2.0**。训练请使用具有相应使用权的兼容输入。

[开始使用](README.md) · [参与贡献](CONTRIBUTING.md)
