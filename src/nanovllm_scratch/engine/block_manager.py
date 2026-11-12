# KV cache 的物理块管理。
# 对应原版 NanoVLLM：nanovllm/engine/block_manager.py。
# 以主仓库记录的参考版本 bb823b3 为接口依据。
# 本文件按实现顺序说明各接口，不表示一次推理的执行顺序。
# Sequence 保存请求的逻辑块信息；BlockManager 管理物理块编号及其占用关系。
# 实际 KV 张量由执行侧管理，本模块不执行模型前向或申请 GPU 张量。
# token_ids 是 list[int]，block_table 是 list[int]，这里没有 batch 张量。
# block_size 必须与 Sequence.block_size 和执行侧 KV cache 的块大小一致。
#
# 第一步：准备本模块需要的类型和工具。
# deque：保存可分配的物理块编号，支持从队首取块、向队尾归还块。
# Sequence：读取请求的 token、逻辑块划分以及计算进度。
# 原版用 xxhash.xxh64 计算哈希，用 NumPy 将 token IDs 编码为字节。
# 当前项目未显式声明 xxhash 和 NumPy；采用这些工具时另行处理依赖。
#
# 第二步：定义 Block，表示一个物理块的元数据。
# __init__(block_id)：输入物理块编号，无业务返回值。
# block_id：物理块的固定编号，块被回收或复用时编号保持不变。
# ref_count：当前引用这个块的请求数，初始为 0。
# hash：这个完整块及其前缀的哈希，初始为 -1，表示尚无可复用身份。
# token_ids：这个块记录的 token ID 列表，初始为空。
# Block 不保存 KV 张量；token_ids 用于核对缓存内容，避免仅凭哈希误判。
#
# 第三步：定义 Block.update(hash: int, token_ids: list[int])。
# 输入：完整块的前缀哈希及该块的 token IDs；无业务返回值。
# 职责：更新块的缓存身份和 token 内容。
# 此方法不改变 block_id 或 ref_count，也不负责向管理器登记哈希。
#
# 第四步：定义 Block.reset()。
# 输入：无；无业务返回值。
# 职责：把空闲块重置为一次新分配的状态。
# 重置后 ref_count 为 1，hash 为 -1，token_ids 为空。
# reset 表示开始新的占用，不是把块归还空闲池。
#
# 第五步：定义 BlockManager.__init__(num_blocks: int, block_size: int)。
# 输入：物理块总数、每个块可容纳的 token 数；无业务返回值。
# block_size：保存统一块大小。
# blocks：按物理编号索引的 Block 列表，共 num_blocks 个。
# hash_to_block_id：dict[int, int]，记录可复用的前缀哈希对应哪个物理块。
# free_block_ids：deque[int]，初始包含全部物理块编号。
# used_block_ids：set[int]，初始为空，记录正在被请求引用的块编号。
# 空闲集合与使用集合互不重叠，两者合起来覆盖全部物理块。
# 空闲块可能还保留旧 KV 和哈希；空闲表示可以重新分配，不等于内容已清除。
#
# 第六步：定义类方法 compute_hash(token_ids: list[int], prefix: int = -1)。
# 输入：当前块的 token IDs，以及前面完整块累计得到的哈希。
# 输出：int，表示当前块连同此前前缀的哈希。
# prefix 为 -1 表示没有前缀；其他值参与本块的哈希计算。
# 原版用 64 位 xxhash：有效前缀编码为 8 字节小端序，再加入 token 字节。
# token 字节来自 NumPy 整数数组；同一运行环境内需保持编码一致。
# 相同块内容放在不同前缀后面，其 KV 通常不同，因此不能只哈希本块 token。
#
# 第七步：定义 _allocate_block() -> int。
# 输入：无；输出：新占用的物理块编号。
# 前提：空闲池非空；取出的块 ref_count 为 0。
# 职责：从空闲队列的队首取得块，使其进入使用集合并重置为新分配状态。
# 旧哈希映射若仍指向此块，需要失效；不能误删已经指向其他块的映射。
# 此方法只取得一个物理块，不修改具体请求的 block_table。
#
# 第八步：定义 _deallocate_block(block_id: int)。
# 输入：待归还的物理块编号；无业务返回值。
# 前提：该块仍在使用集合，但 ref_count 已降为 0。
# 职责：把块移出使用集合，加入空闲队列的队尾。
# 归还时保留 hash、token_ids 和可用哈希映射，允许后续请求复用旧缓存。
# 与 _allocate_block 的重置行为区分：尚未重新占用时，缓存仍可能有效。
#
# 第九步：定义 can_allocate(seq: Sequence) -> int。
# 输入：尚未分配物理块、block_table 为空的请求。
# 输出：空间不足返回 -1；空间足够返回可复用的连续前缀块数，可以为 0。
# 原版这个接口返回 int，不能把它当作 bool；0 表示可以分配但没有缓存命中。
# 只查询，不修改块池、引用计数、请求块表或 num_cached_tokens。
# 仅考虑最后一个逻辑块之前的连续完整块；原版即使末块已满，也不复用末块。
# 前缀中途未命中或 token 内容不匹配，后面的块不再计入可复用前缀。
# 命中的使用中块可以共享，不消耗空闲名额。
# 命中的空闲块仍需重新占用，必须计入空闲块需求。
# 尚未命中的各逻辑块也需要空闲块；所有需求必须能由当前空闲池满足。
#
# 第十步：定义 allocate(seq: Sequence, num_cached_blocks: int)。
# 输入：未分配的请求，以及 can_allocate 刚返回的非负前缀块数。
# 前提：seq.block_table 为空，查询后没有其他操作改变分配条件。
# 输出：无业务返回值；请求获得与逻辑块顺序一致的物理 block_table。
# 对前缀命中块建立本请求的引用；使用中块增加引用，空闲缓存块重新进入使用状态。
# 复用空闲缓存块时保留其身份与内容，不能按全新分配重置。
# 剩余逻辑块获得新占用的物理块。
# seq.num_cached_tokens 表示已复用的完整前缀长度，不把新分配空间算作已有 KV。
# 此方法不执行前向，也不修改请求的 token 内容、调度状态或本轮计算预算。
#
# 第十一步：定义 deallocate(seq: Sequence)。
# 输入：待释放资源的请求；无业务返回值。
# 职责：释放该请求对 block_table 中各物理块的引用。
# 只有引用数降到 0 的块才能回到空闲池；仍被其他请求共享的块继续占用。
# 原版按 block_table 的逆序归还，配合空闲队列保留较早前缀的复用机会。
# 完成后 seq.block_table 为空，seq.num_cached_tokens 为 0。
# 保留请求的 token、生成参数和身份；是否结束或抢占由 Scheduler 决定。
#
# 第十二步：定义 can_append(seq: Sequence) -> bool。
# 输入：已经有块表，且刚追加了一个新 token 的运行中请求。
# 输出：bool，表示下一次 Decode 是否有足够物理空间容纳该 token 的 KV。
# 只查询，不分配、不改变引用计数或请求元数据。
# 这里的序列长度已经包含新 token；它的 KV 尚未执行前向。
# 新 token 落入已有末块时不需要新块；落入新逻辑块时需要一个空闲块。
# 原版此处针对每次追加一个 token 的 Decode，不是任意长度批量扩展接口。
#
# 第十三步：定义 may_append(seq: Sequence)。
# 输入：满足 can_append 的请求，调用时机与 can_append 相同。
# 输出：无业务返回值；必要时为新增逻辑块扩充 block_table。
# 已有末块能容纳该 token 时保持块表不变。
# 此方法不追加 token、不增加 num_cached_tokens，也不登记完整块哈希。
#
# 第十四步：定义 hash_blocks(seq: Sequence)。
# 输入：本轮模型计算已完成、计算进度尚未结算的请求；无业务返回值。
# seq.num_cached_tokens 是本轮之前已有 KV 的前缀长度。
# seq.num_scheduled_tokens 是本轮实际安排计算的 token 数。
# 职责：为本轮计算后新形成的完整块更新 Block 身份，并登记 hash_to_block_id。
# 只登记完整块，不登记仍未填满的尾块，也不登记尚未执行前向的新生成 token。
# 新块哈希应衔接前一个完整块的哈希，保持整段前缀身份连续。
# 若本轮没有新形成完整块，不需要更新缓存登记。
# 此方法不推进 num_cached_tokens，也不清零 num_scheduled_tokens；这些由 Scheduler 结算。
#
# 调用关系：Scheduler 在接纳请求时查询并分配块，在 Decode 前确保追加空间，
# 在计算完成后登记完整块，在请求完成或被抢占时释放该请求的引用。
# BlockManager 不负责等待队列、采样、EOS 判断或请求状态转换。
