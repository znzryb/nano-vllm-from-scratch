import torch
from torch._C import dtype
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
        # 形状为 [1, seq]：1 表示单请求，seq 表示当前序列长度。
        # 当前不接收字符串，不处理 padding。

        # 输出约定：
        # 返回 logits 张量，形状为 [1, seq, vocab]。
        # vocab 是词表大小，d_model 是隐藏维度。
        # logits 是未经 softmax 的分数。

        # 第一步：检查输入。
        # 确认是二维张量，batch size 为 1，dtype 为 torch.long。
        # 确认 1 <= seq <= self.max_sequence_len。
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
        # 默认保留全部 seq 个位置，结果形状应为 [1, seq, vocab]。

        # 第五步：返回 logits。
        # 此处不做 softmax、argmax、采样或文本解码。
        # 选择下一个 token、拼接序列和判断 EOS，留给 generate。
        return outputs.logits

    @torch.inference_mode()
    def generate(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 128,
    ) -> list[int]:
        # 职责：
        # 根据 prompt，逐个生成后续 token。
        # 每次选择预测分数最高的 token，即 greedy decoding。

        # 输入约定：
        # input_ids：prompt 的 token IDs，形状 [1, seq]，dtype 为 torch.long。
        # max_new_tokens：最多新增多少个 token，不包含 prompt 的长度。

        # 输出约定：
        # 返回新增 token IDs 的 Python 列表，不包含 prompt。
        # 如果生成了 EOS，返回列表包含这个 EOS。
        # 不在这里将 token IDs 解码成文本。

        # 第一步：检查参数。
        # 对 input_ids 做与 forward 一致的合法性检查。
        # max_new_tokens 必须是非负整数。
        # 本版采用明确的长度策略：
        # prompt 长度 + max_new_tokens 不得超过 self.max_sequence_len。
        # 超出就报错，不自动截断 prompt 或减少生成数量。
        if isinstance(max_new_tokens, bool):
            raise TypeError(
                f"Expected max_new_tokens to be int, got {type(max_new_tokens)}"
            )
        if not isinstance(max_new_tokens, int):
            raise TypeError(
                f"Expected max_new_tokens to be int, got {type(max_new_tokens)}"
            )
        if not max_new_tokens >= 0:
            raise ValueError(
                f"Expected max_new_tokens to be >= 0, got max_new_tokens={max_new_tokens}"
            )

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

        if not input_ids.shape[-1] + max_new_tokens <= self.max_sequence_len:
            raise ValueError(
                f"Expected prompt_len(input_ids.shape[-1]) + max_new_tokens to be <= self.max_sequence_len, "
                f"got prompt_len(input_ids.shape[-1]) + max_new_tokens = {input_ids.shape[-1] + max_new_tokens} "
                f"self.max_sequence_len:{self.max_sequence_len}"
            )

        if input_ids.shape[0] != 1:
            raise ValueError(f"Expected input_ids to be 1 at first dimension (batch_size), "
                             f"got {input_ids.shape[0]} at first dimension (batch_size)")
        if not 1 <= input_ids.shape[-1] <= self.max_sequence_len:
            raise ValueError(f"Expected input_ids.shape[-1] to be 1 <= input_ids.shape[-1] <= {self.max_sequence_len}, "
                             f"got  input_ids.shape[-1]:{input_ids.shape[-1]}")
        if torch.max(input_ids) >= self.config.vocab_size or torch.min(input_ids) < 0:
            raise ValueError(f"Expected input_ids element (token_id) to be [0, {self.config.vocab_size}), "
                             f"got min:{torch.min(input_ids)} max:{torch.max(input_ids)}")

        # 第二步：处理不需要生成的情况。
        # 如果 max_new_tokens 为 0，返回空列表。
        # 此时不需要执行模型前向。
        if max_new_tokens == 0:
            return []
        # 第三步：准备本次请求的状态。
        # 将 prompt 移到 self.device，保持 torch.long。
        # 用一个局部变量保存“当前完整序列”，起初只有 prompt。
        # 创建一个局部列表，保存新增 token IDs。
        # 不把这些请求状态保存在 self 上，避免影响下一次调用。
        input_ids = input_ids.to(device=self.device)
        # 第四步：开始生成循环，最多执行 max_new_tokens 次。
        # 以下第五步到第八步，都放在循环内部。
        new_token_ls = []
        for i in range(max_new_tokens):
            # 第五步：预测下一个 token。
            # 将“当前完整序列”传给 self.forward，得到 [1, seq, vocab]。
            # 无缓存模式每次都要传完整序列，不能只传最后一个 token。
            # 取最后一个位置的 logits，得到 [vocab]。（vocab_size）
            # 最后一个位置的输出，表示对下一个 token 的预测。
            logits = self.forward(input_ids)
            logits = logits[*(size - 1 for size in logits.shape[:-2]), logits.shape[-2] - 1]
            # logits.unsqueeze(-1)
            # 第六步：按 greedy 策略选择 token。
            # 沿词表维度做 argmax，得到分数最高的 token ID。
            # 不需要先做 softmax，因为 softmax 不改变最大值的位置。
            # 将这个 ID 转成 Python int，加入新增 token 列表。
            # 这个 argmax 直接返回索引，.item 用来转化为这个 python 整数
            # >>> x = torch.tensor([1.0])
            # >>> x.item()
            # 1.0
            token_id = torch.argmax(logits).item()

            # 第七步：检查 EOS。
            # 如果 self.eos_token_id 不为 None，
            # 且刚生成的 token 等于它，结束循环。
            # 注意先保存 token，再判断停止，这样结果中包含 EOS。
            # 不因 prompt 本身以 EOS 结尾就直接停止。
            new_token_ls.append(token_id)
            if self.eos_token_id is not None:
                if token_id == self.eos_token_id:
                    break

            if len(new_token_ls) >= max_new_tokens:
                break
            # 第八步：更新下一轮的输入。
            # 如果还要继续生成，将新 token 接到当前完整序列末尾。
            # 新 token 张量需要是 [1, 1]，保持相同设备和 torch.long。
            # 沿序列维度拼接，完整序列由 [1, seq] 变成 [1, seq + 1]。
            # 拼接后的结果要保存，供下一轮 forward 使用。
            input_ids = torch.cat((input_ids,
                                   torch.tensor([[token_id]], dtype=torch.long, device=self.device))
                                  , dim=1)  # dim 就是想要变长的那个维度

        # 第九步：循环结束后返回新增 token 列表。
        # 结束原因可能是遇到 EOS，或者达到 max_new_tokens。
        # 返回列表长度不超过 max_new_tokens。
        return new_token_ls
