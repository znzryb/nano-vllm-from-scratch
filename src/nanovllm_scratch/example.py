# 导入 argparse，用于解析命令行参数。
import argparse

# 导入 torch，用于将精度名称映射为 torch.dtype。
import torch

# 从 nanovllm_scratch.baseline 导入 BaseLine。
from nanovllm_scratch.baseline import BaseLine


# 定义模块级函数 main()，成功时返回整数 0。
def main() -> int:
    #
    # 第一步：创建 argparse.ArgumentParser，描述程序用途。
    argument_parser = argparse.ArgumentParser()
    #
    # 第二步：注册命令行参数。
    # --model-path：
    #   字符串，必填，表示本地模型与 tokenizer 所在目录。
    argument_parser.add_argument("--model-path", type=str, required=True)
    #
    # --prompt：
    #   字符串，必填，表示需要续写的原始文本。
    argument_parser.add_argument("--prompt", type=str, required=True)
    #
    # --max-new-tokens：
    #   整数，建议默认 128，表示最多新增多少个 token。
    argument_parser.add_argument("--max-new-tokens", type=int, default=128)
    #
    # --device：
    #   字符串，建议默认 "cuda"；测试会显式传入 "cpu"。
    argument_parser.add_argument("--device", type=str, default="cuda")
    #
    # --dtype：
    #   字符串，允许 "float32" 和 "bfloat16"。
    #   建议默认 "bfloat16"；测试会显式传入 "float32"。
    argument_parser.add_argument(
        "--dtype", type=str, choices=["bfloat16", "float32"], default="bfloat16"
    )
    #
    # 第三步：解析参数。
    # argparse 默认支持 --help，会打印帮助并正常退出。
    # 必须先解析参数，再加载模型，确保 --help 不依赖模型文件。
    # argument_parser.add_argument("--help")
    args = argument_parser.parse_args()
    #
    # 第四步：检查生成预算。
    # 如果 max_new_tokens 小于 0，调用 parser.error 报错。
    # 错误消息包含 "--max-new-tokens"，指出它必须大于等于 0。
    # parser.error 会向 stderr 输出错误，并以非零退出码结束程序。
    if args.max_new_tokens < 0:
        argument_parser.error(
            f"Expected --max-new-tokens >= 0, got max-new-tokens={args.max_new_tokens}"
        )
    #
    # 第五步：转换精度参数。
    # 将 "float32" 映射为 torch.float32。
    # 将 "bfloat16" 映射为 torch.bfloat16。
    # BaseLine 的 dtype 参数需要 torch.dtype，不能直接传字符串。
    if args.dtype == "float32":
        torch_type = torch.float32
    else:
        torch_type = torch.bfloat16
    #
    # 第六步：创建 BaseLine 实例。
    # 将解析得到的 model_path、device 和转换后的 dtype 传给构造函数。
    # 已有构造函数只加载本地文件，不自动下载模型。
    baseline_model = BaseLine(args.model_path, args.device, torch_type)
    #
    # 第七步：将 prompt 编码成 token IDs。
    # 使用实例的 tokenizer，指定 return_tensors="pt"。
    # return_tensors="pt" 表示将编码结果中的数值字段返回为 PyTorch 张量。
    # 从返回结果中取出 "input_ids"。
    # 得到 torch.long 张量，形状为 [1, seq]。
    # 本例直接编码原始文本，不添加聊天模板，不做 padding 或截断。
    # generate 内部会负责将张量移动到目标设备。 
    # tokenizer() 的返回值中有 input_ids，也有这个 attention_mask
    input_ids = baseline_model.tokenizer(args.prompt, return_tensors="pt").input_ids
    #
    # 第八步：调用实例的 generate。
    # 传入 input_ids 和 max_new_tokens。
    # 返回值是新增 token IDs 的 Python 列表，不包含 prompt。
    # EOS、上下文长度及零生成预算由已有 generate 处理。
    new_token_ids = baseline_model.generate(input_ids, args.max_new_tokens)
    #
    # 第九步：将新增 token IDs 解码成文本。
    # 使用同一个 tokenizer 的 decode，指定 skip_special_tokens=True。
    # generate 已经只返回新增 token，因此不要再按 prompt 长度切片。
    # 新增 token 列表为空时，解码结果应为空字符串。
    text_gen = baseline_model.tokenizer.decode(new_token_ids,skip_special_tokens=True)
    #
    # 第十步：输出结果。
    # stdout 只打印解码得到的文本，允许末尾一个换行。
    # 不打印 prompt、参数、token 列表或“生成结果：”等前缀。
    # 返回整数 0。
    print(text_gen)
    return 0


# 模块入口：
# 当 __name__ == "__main__" 时调用 main()。
# 使用 raise SystemExit(main()) 将返回值作为进程退出码。
# 这样即可通过 python -m nanovllm_scratch.example 运行。
if __name__ == "__main__":
    raise SystemExit(main())