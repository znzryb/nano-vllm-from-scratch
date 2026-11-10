---
paths:
  - "**"
---

# 张量形状与术语

- 项目注释、文档、测试说明和讲解统一使用 `B / S / V / D` 表示张量各轴的长度：`B` 为 batch size（批大小），`S` 为 sequence length（序列长度），`V` 为 vocabulary size（词表大小），`D` 为 hidden size（隐藏维度，也称 `d_model`）。
- 序列长度统一写 `S`，不混用 `T`、`L` 或 `Seq_Len`；`V` 和 `D` 不表示序列长度。已有语义清楚的代码标识符（如 `max_sequence_len`、`hidden_size`）不必改名。
- 常见形状：token IDs 为 `[B, S]`，logits 为 `[B, S, V]`，隐藏状态为 `[B, S, D]`。当前单请求 baseline 中 `B = 1`；`S` 表示当前实际序列长度，最大允许长度另行说明。
- 讲解时区分张量的维数（rank）、某一轴的长度和元素的值；例如 `[B, S, D]` 是三维张量，`D` 是最后一轴的长度，不是张量的维数。
- 引用原版 NanoVLLM 或第三方源码时保留其真实标识符，并解释与本项目记号的对应关系；不为统一记号修改 `references/` 子模块。
