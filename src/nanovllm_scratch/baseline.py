import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM


class BaseLine:
    def __init__(self,
                 model_path,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16,
                 ):
        self.device = device
        self.dtype = dtype

        self.model = Qwen3ForCausalLM.from_pretrained(
            model_path,
            # 不自动下载
            local_files_only=True,
            dtype=dtype
        )

        # 把模型参数移动到指定设备。
        self.model.to(self.device)

        # 切换到推理模式，关闭 dropout 等训练行为。
        # “评估”，就是：让模型用已经学到的参数计算结果，检查它表现得怎么样
        self.model.eval()  # evaluation

        # baseline 不训练，模型参数不需要梯度。
        self.model.requires_grad_(False)

        # use_fast=True 表示：优先使用 Rust 实现的快速版 tokenizer。
        # use_fast=False	使用 Python 慢速版
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            local_files_only=True,
        )

        self.config = self.model.config

        self.max_sequence_len = min(
            4096,
            self.config.max_position_embeddings
        )
        # 保存结束 token，后续生成逻辑用它判断是否停止。
        self.eos_token_id = self.tokenizer.eos_token_id

        # 把模型默认使用 KV cache 的设置关闭
        self.config.use_cache = False

    @torch.inference_mode()  # 本次前向不记录梯度。
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 职责：
        # 接收当前完整序列的 token IDs，返回每个位置的词表预测分数。

        # 输入约定：
        # input_ids 是 torch.long 类型的整数张量。
        # 形状为 [1, S]：1 表示单请求，S 表示当前序列长度。
        # 当前不接收字符串，不处理 padding。

        # 输出约定：
        # 返回 logits 张量，形状为 [1, S, V]。
        # V 是词表大小，不是 hidden_size。
        # logits 是未经 softmax 的分数。

        # 第一步：检查输入。
        # 确认是二维张量，batch size 为 1，dtype 为 torch.long。
        # 确认 1 <= S <= self.max_sequence_len。
        # token ID 必须在 [0, self.config.vocab_size) 范围内。
        # 不合法时明确报错，不默默截断或转换浮点输入。
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                f"Expected input_ids to be a torch.Tensor, got {type(input_ids)}"
            )
        if input_ids.dtype != torch.long:
            raise TypeError(f"Expected input_ids.dtype to be torch.long, "
                            f"got {input_ids.dtype}")
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids to be 2 dimensions, "
                             f"got {input_ids.ndim} dimensions")
        if input_ids.shape[0] != 1:
            raise ValueError(f"Expected input_ids to be 1 at first dimension (batch_size), "
                             f"got {input_ids.shape[0]} at first dimension (batch_size)")
        if not 1 <= input_ids.shape[-1] <= self.max_sequence_len:
            raise ValueError(f"Expected input_ids.shape[-1] to be 1 <= input_ids.shape[-1] <= {self.max_sequence_len}, "
                             f"got  input_ids.shape[-1]:{input_ids.shape[-1]}")
        if torch.max(input_ids) >= self.config.vocab_size or torch.min(input_ids) < 0:
            raise ValueError(f"Expected input_ids element (token_id) to be [0, {self.config.vocab_size}), "
                             f"got min:{torch.min(input_ids)} max:{torch.max(input_ids)}")

        # 第二步：准备设备。
        # 将 input_ids 移到 self.device，使输入和模型位于同一设备。
        # 保持整数类型，不要转换成模型权重使用的 bfloat16。
        input_ids = input_ids.to(device=self.device)
        # 第三步：执行模型前向。
        # 调用 self.model(...)，通过 input_ids 参数传入输入。
        # 显式传入 use_cache=False，保持无缓存对照路径。
        # 显式传入 return_dict=True，方便通过属性读取结果。
        # 不传 labels：这里计算预测分数，不计算训练 loss。
        # 对当前无 padding 的完整单序列，位置和因果掩码由模型处理。
        outputs = self.model(input_ids, use_cache=False, return_dict=True)
        # 第四步：提取 logits。
        # 从模型返回对象的 .logits 属性读取预测分数。
        # 默认保留全部 S 个位置，结果形状应为 [1, S, V]。

        # 第五步：返回 logits。
        # 此处不做 softmax、argmax、采样或文本解码。
        # 选择下一个 token、拼接序列和判断 EOS，留给 generate。
        return outputs.logits
