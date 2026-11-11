# 推理请求的序列状态管理。
# 对应原版 NanoVLLM：nanovllm/engine/sequence.py。
# 本文件按实现顺序列出练习要求；各步骤是不同的接口，不是一次生成的执行顺序。
# token IDs 使用 Python 整数列表；这里没有 batch 张量。
# Sequence 保存请求状态和缓存元数据，实际 KV 张量由执行侧管理。
#
# 第一步：定义 SequenceStatus。
# 用枚举表示三个状态：WAITING、RUNNING、FINISHED。
# WAITING：等待调度；RUNNING：已进入运行队列；FINISHED：请求已结束。
# Sequence 提供状态记录，具体调度和结束判定由引擎相关模块负责。
#
# 第二步：定义 Sequence 的类级配置。
# block_size：每个逻辑块最多容纳多少个 token，原版此处为 256。
# 后续接入缓存时，它必须与 BlockManager 和实际 KV cache 的块大小一致。
# counter：所有 Sequence 实例共用的请求编号来源。
# 检查点：连续创建两条请求，它们的 seq_id 应不同。
#
# 第三步：实现构造函数 __init__。
# 输入：token_ids 为非空的 prompt token ID 列表。
# 输入：sampling_params 为生成参数，对应原版的 SamplingParams。
# 原版参数类型位于 nanovllm/sampling_params.py；本项目尚未建立对应模块。
# 输出：构造函数不返回业务结果；实例应具有下面这些初始状态。
#
# 3.1 请求身份与状态。
# seq_id：当前请求的唯一编号。
# status：初始状态为 WAITING。
#
# 3.2 token 内容与长度。
# token_ids：当前完整序列，最初只有 prompt；与调用者的列表相互独立。
# last_token：当前序列的最后一个 token ID。
# num_tokens：当前完整序列的 token 数。
# num_prompt_tokens：原始 prompt 的 token 数，后续生成时保持不变。
# 检查点：修改调用者原来的列表，不应改变请求中保存的 token 内容。
#
# 3.3 计算进度与缓存元数据。
# num_cached_tokens：已有可用 KV cache 的前缀 token 数，初始为 0。
# num_scheduled_tokens：本轮安排计算的 token 数，初始为 0。
# is_prefill：是否走 Prefill 路径，初始为 True。
# block_table：逻辑块对应的物理 KV cache 块编号，初始为空列表。
# 注意：逻辑上需要几个块，与实际已经分配几个物理块是两个概念。
#
# 3.4 本请求的生成参数。
# temperature：采样温度。
# max_tokens：最多生成多少个新 token，不包含 prompt。
# ignore_eos：是否忽略 EOS 停止条件。
# 这里只保存参数，采样和停止判定在其他模块执行。
#
# 第四步：实现基本访问接口。
# __len__：返回当前完整序列的 token 数，结果为 int。
# __getitem__：支持按索引访问 token，也支持按切片访问一段 token。
# 输入：整数索引或切片；输出：单个 int 或 list[int]。
# 检查点：刚创建时，长度与 prompt 长度相同，索引访问与保存的列表一致。
#
# 第五步：实现请求状态与 token 分段属性。
# 以下属性均为读取接口，不修改请求状态。
# is_finished：返回 bool，表示 status 是否为 FINISHED。
# num_completion_tokens：返回已经生成的新 token 数，不包含 prompt。
# prompt_token_ids：返回原始 prompt 对应的 token ID 列表。
# completion_token_ids：返回新增 token ID 列表，不包含 prompt。
# 检查点：刚创建时，新增 token 数为 0，新增 token 列表为空。
# 检查点：完整 token 内容由 prompt 和新增 token 两部分按顺序组成。
#
# 第六步：实现逻辑块的只读接口。
# num_blocks：当前完整序列所需的逻辑块数，返回 int。
# last_block_num_tokens：最后一个逻辑块包含的 token 数，返回 int。
# block(i)：返回第 i 个逻辑块对应的 token ID 列表，逻辑块从 0 编号。
# block(i) 的输入必须落在有效逻辑块编号范围内。
# block(i) 返回 token 内容；block_table 保存物理块编号，两者用途不同。
# 检查点：长度分别为 1、block_size、block_size + 1 时，块划分正确。
# 检查点：序列长度恰好为 block_size 时，最后一块包含 block_size 个 token。
# 检查点：按顺序拼接所有逻辑块，能还原完整 token ID 列表。
#
# 第七步：实现 append_token(token_id)。
# 输入：新生成的单个 token ID，类型为 int。
# 输出：无业务返回值；请求中保存的完整序列增长一个 token。
# 调用后，token_ids、last_token 和 num_tokens 应保持一致。
# num_prompt_tokens 保持不变，新增 token 数相应增加。
# 新 token 尚未执行模型前向，不应仅因追加就算作已有 KV cache 的 token。
# 此方法不负责采样、分配物理块或判定请求结束。
# 检查点：追加前后，原始 prompt 内容不变，新增部分按追加顺序增长。
#
# 第八步：后续接入多进程 ModelRunner 时，实现状态传输接口。
# 单进程阶段可以先完成前七步，再处理这一部分。
# 原版使用 __getstate__ 和 __setstate__ 配合 Python 对象序列化。
# 它们传输的是执行侧需要的精简状态，不是完整请求的持久化备份。
#
# __getstate__：无额外输入，返回供序列化使用的状态元组。
# 原版元组依次包含：num_tokens、num_prompt_tokens、num_cached_tokens、
# num_scheduled_tokens、block_table，以及本次执行需要的 token 信息。
# Prefill 路径传完整 token 列表；Decode 路径传 last_token。
#
# __setstate__(state)：接收上述元组，恢复执行侧需要的字段，无业务返回值。
# Prefill 状态恢复 token 列表与最后一个 token。
# Decode 状态仅保留最后一个 token，原版此时将 token_ids 设为空列表。
# 因此，恢复后的 Decode 对象不具备完整历史，不能用于查询完整生成文本。
# 检查点：两种路径恢复后，计数、块表和执行所需 token 信息均与传输前一致。
#
# 最终验收：创建请求、读取属性、追加 token、跨块边界访问均符合上述约定。
# 这些状态逻辑可以在 CPU 上验证，不需要模型权重或 GPU。
