---
paths:
  - "**"
---

# 张量形状与术语

- 项目注释、文档、测试说明和讲解统一使用清楚的维度名称：`batch` 为 batch size（批大小），`seq` 为 sequence length（序列长度），`vocab` 为 vocabulary size（词表大小），`d_model` 为 hidden size（隐藏维度）。
- 形状说明优先写完整名称，不用 `B / S / V / D` 等单字母缩写；序列长度统一写 `seq`，不混用 `T`、`L` 或 `Seq_Len`。已有语义清楚的代码标识符（如 `max_sequence_len`、`vocab_size`、`hidden_size`）不必改名。
- 常见形状：token IDs 为 `[batch, seq]`，logits 为 `[batch, seq, vocab]`，隐藏状态为 `[batch, seq, d_model]`。当前单请求 baseline 中 `batch = 1`，可以写 `[1, seq, vocab]`；`seq` 表示当前实际序列长度，最大允许长度另行说明。
- 讲解时区分张量的维数（rank）、某一轴的长度和元素的值；例如 `[batch, seq, d_model]` 是三维张量，`d_model` 是最后一轴的长度，不是张量的维数。
- 引用原版 NanoVLLM 或第三方源码时保留其真实标识符，并解释与本项目记号的对应关系；不为统一记号修改 `references/` 子模块。
