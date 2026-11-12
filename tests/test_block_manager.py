"""BlockManager 的 CPU 行为测试，对齐参考版本 bb823b3 的公开接口。

运行：uv run pytest tests/test_block_manager.py -q
不加载模型，不执行 GPU 运算，不在测试中提供块管理的参考实现。
延后访问待实现类型，让其他测试仍可收集；接口缺失时正常失败。
"""

from importlib import import_module

import pytest

from nanovllm_scratch.engine.sequence import Sequence


@pytest.fixture
def block_module(monkeypatch):
    # 缩小逻辑块，便于用少量 token 观察物理块分配与共享。
    monkeypatch.setattr(Sequence, "block_size", 4)
    return import_module("nanovllm_scratch.engine.block_manager")


def assert_pool_consistent(manager):
    free_ids = list(manager.free_block_ids)
    used_ids = set(manager.used_block_ids)
    assert len(free_ids) == len(set(free_ids))
    assert set(free_ids).isdisjoint(used_ids)
    assert set(free_ids) | used_ids == set(range(len(manager.blocks)))
    for block_id in free_ids:
        assert manager.blocks[block_id].ref_count == 0
    for block_id in used_ids:
        assert manager.blocks[block_id].ref_count > 0


@pytest.fixture
def cached_request(block_module):
    manager = block_module.BlockManager(3, 4)
    seq = Sequence(list(range(8)))
    manager.allocate(seq, manager.can_allocate(seq))
    seq.num_scheduled_tokens = 8
    manager.hash_blocks(seq)
    # 模拟调用方在前向完成后的进度结算，不提供缓存管理算法。
    seq.num_cached_tokens = 8
    seq.num_scheduled_tokens = 0
    return manager, seq


def test_block_initial_metadata(block_module):
    block = block_module.Block(7)
    assert block.block_id == 7
    assert block.ref_count == 0
    assert block.hash == -1
    assert block.token_ids == []


def test_block_update_preserves_identity_and_reference_count(block_module):
    block = block_module.Block(7)
    block.ref_count = 2
    assert block.update(123, [10, 11, 12, 13]) is None
    assert block.hash == 123
    assert block.token_ids == [10, 11, 12, 13]
    assert block.block_id == 7
    assert block.ref_count == 2


def test_block_reset_starts_new_ownership(block_module):
    block = block_module.Block(7)
    block.update(123, [10, 11, 12, 13])
    assert block.reset() is None
    assert block.block_id == 7
    assert block.ref_count == 1
    assert block.hash == -1
    assert block.token_ids == []


def test_manager_initial_pool(block_module):
    manager = block_module.BlockManager(3, 4)
    assert manager.block_size == 4
    assert [block.block_id for block in manager.blocks] == [0, 1, 2]
    assert list(manager.free_block_ids) == [0, 1, 2]
    assert manager.used_block_ids == set()
    assert manager.hash_to_block_id == {}
    assert_pool_consistent(manager)


def test_hash_is_repeatable_and_depends_on_tokens_and_prefix(block_module):
    compute = block_module.BlockManager.compute_hash
    original = compute([10, 11, 12, 13])
    assert isinstance(original, int)
    assert 0 <= original < 2**64
    assert compute([10, 11, 12, 13], -1) == original
    assert compute([10, 11, 12, 13]) == original
    assert compute([10, 11, 12, 99]) != original
    assert compute([10, 11, 12, 13], original) != original
    assert compute([10, 11, 12, 13], 123) != compute([10, 11, 12, 13], 456)


@pytest.mark.parametrize("length,expected_blocks", [(1, 1), (4, 1), (5, 2), (8, 2), (9, 3)])
def test_new_request_gets_required_physical_blocks(block_module, length, expected_blocks):
    manager = block_module.BlockManager(3, 4)
    seq = Sequence(list(range(length)))
    assert manager.can_allocate(seq) == 0
    assert manager.allocate(seq, 0) is None
    assert len(seq.block_table) == expected_blocks
    assert len(set(seq.block_table)) == expected_blocks
    assert set(seq.block_table) == manager.used_block_ids
    assert seq.num_cached_tokens == 0
    assert all(manager.blocks[block_id].ref_count == 1 for block_id in seq.block_table)
    assert seq.token_ids == list(range(length))
    assert_pool_consistent(manager)


def test_allocation_query_does_not_mutate_state(block_module):
    manager = block_module.BlockManager(1, 4)
    seq = Sequence(list(range(5)))
    assert manager.can_allocate(seq) == -1
    assert manager.can_allocate(seq) == -1
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
    assert list(manager.free_block_ids) == [0]
    assert manager.used_block_ids == set()
    assert manager.hash_to_block_id == {}
    assert_pool_consistent(manager)


def test_allocation_is_not_cache_publication(block_module):
    manager = block_module.BlockManager(2, 4)
    seq = Sequence(list(range(8)))
    manager.allocate(seq, 0)
    assert manager.hash_to_block_id == {}
    assert all(manager.blocks[block_id].hash == -1 for block_id in seq.block_table)


def test_active_prefix_is_shared_but_final_full_block_is_private(cached_request):
    manager, first = cached_request
    second = Sequence(list(range(8)))
    assert manager.can_allocate(second) == 1
    assert manager.blocks[first.block_table[0]].ref_count == 1
    assert second.block_table == []
    manager.allocate(second, 1)
    assert second.block_table[0] == first.block_table[0]
    assert second.block_table[1] != first.block_table[1]
    assert manager.blocks[first.block_table[0]].ref_count == 2
    assert second.num_cached_tokens == 4
    assert first.num_cached_tokens == 8
    assert_pool_consistent(manager)


def test_single_full_block_is_not_reused(cached_request):
    manager, _ = cached_request
    seq = Sequence([0, 1, 2, 3])
    assert manager.can_allocate(seq) == 0


def test_free_cached_prefix_still_requires_free_capacity(block_module):
    manager = block_module.BlockManager(2, 4)
    first = Sequence(list(range(8)))
    manager.allocate(first, 0)
    first.num_scheduled_tokens = 8
    manager.hash_blocks(first)
    manager.deallocate(first)
    # 命中一个空闲缓存块，另有两个逻辑块需要空间；总需求仍然是三个。
    candidate = Sequence(list(range(9)))
    assert manager.can_allocate(candidate) == -1
    assert_pool_consistent(manager)


def test_free_cached_prefix_reactivates_without_resetting_content(cached_request):
    manager, first = cached_request
    prefix_id = first.block_table[0]
    prefix_hash = manager.blocks[prefix_id].hash
    manager.deallocate(first)
    second = Sequence(list(range(8)))
    assert manager.can_allocate(second) == 1
    manager.allocate(second, 1)
    block = manager.blocks[prefix_id]
    assert second.block_table[0] == prefix_id
    assert block.ref_count == 1
    assert block.hash == prefix_hash
    assert block.token_ids == [0, 1, 2, 3]
    assert second.num_cached_tokens == 4
    assert_pool_consistent(manager)


def test_prefix_miss_prevents_later_block_reuse(cached_request):
    manager, first = cached_request
    manager.deallocate(first)
    # 中间块 token 相同，但之前的上下文不同，不能独立命中。
    candidate = Sequence([99, 1, 2, 3, 4, 5, 6, 7, 8])
    assert manager.can_allocate(candidate) == 0


def test_hash_candidate_must_also_match_token_content(cached_request):
    manager, first = cached_request
    prefix_hash = manager.blocks[first.block_table[0]].hash
    # 人为注入哈希候选内容不匹配，验证不能仅凭字典命中复用。
    manager.hash_to_block_id[prefix_hash] = first.block_table[1]
    assert manager.can_allocate(Sequence(list(range(8)))) == -1
    # 空间不足掩盖了返回的命中数量；释放一块后应明确返回零命中。
    manager.deallocate(first)
    assert manager.can_allocate(Sequence(list(range(8)))) == 0


def test_shared_block_is_freed_only_after_last_request(cached_request):
    manager, first = cached_request
    second = Sequence(list(range(8)))
    manager.allocate(second, manager.can_allocate(second))
    shared_id = first.block_table[0]
    first_tokens = first.token_ids.copy()
    first_id = first.seq_id
    assert manager.deallocate(first) is None
    assert first.block_table == []
    assert first.num_cached_tokens == 0
    assert first.token_ids == first_tokens
    assert first.seq_id == first_id
    assert manager.blocks[shared_id].ref_count == 1
    assert shared_id in manager.used_block_ids
    assert_pool_consistent(manager)
    manager.deallocate(second)
    assert second.block_table == []
    assert second.num_cached_tokens == 0
    assert len(manager.free_block_ids) == 3
    assert manager.used_block_ids == set()
    assert_pool_consistent(manager)


def test_deallocation_preserves_cached_identity(cached_request):
    manager, seq = cached_request
    block_ids = seq.block_table.copy()
    hashes = [manager.blocks[block_id].hash for block_id in block_ids]
    manager.deallocate(seq)
    for block_id, block_hash in zip(block_ids, hashes):
        assert manager.hash_to_block_id[block_hash] == block_id
        assert manager.blocks[block_id].hash == block_hash
        assert manager.blocks[block_id].token_ids
    assert_pool_consistent(manager)


@pytest.mark.parametrize("length,has_capacity", [(2, True), (3, True), (4, False)])
def test_append_query_and_existing_tail_capacity(block_module, length, has_capacity):
    manager = block_module.BlockManager(1, 4)
    seq = Sequence(list(range(length)))
    manager.allocate(seq, 0)
    seq.append_token(99)
    original_table = seq.block_table.copy()
    assert manager.can_append(seq) is has_capacity
    assert seq.block_table == original_table
    assert_pool_consistent(manager)
    if has_capacity:
        assert manager.may_append(seq) is None
        assert seq.block_table == original_table
        assert seq.num_cached_tokens == 0
        assert seq.token_ids == list(range(length)) + [99]


def test_append_across_boundary_allocates_exactly_one_block(block_module):
    manager = block_module.BlockManager(2, 4)
    seq = Sequence([0, 1, 2, 3])
    manager.allocate(seq, 0)
    original_id = seq.block_table[0]
    seq.num_cached_tokens = 4
    seq.append_token(99)
    assert manager.can_append(seq) is True
    manager.may_append(seq)
    assert len(seq.block_table) == 2
    assert seq.block_table[0] == original_id
    assert seq.block_table[1] != original_id
    assert seq.num_cached_tokens == 4
    assert seq.num_tokens == 5
    assert_pool_consistent(manager)


def test_hash_blocks_publishes_only_completed_blocks(block_module):
    manager = block_module.BlockManager(2, 4)
    seq = Sequence(list(range(7)))
    manager.allocate(seq, 0)
    seq.num_scheduled_tokens = 7
    assert manager.hash_blocks(seq) is None
    first_id, tail_id = seq.block_table
    first = manager.blocks[first_id]
    assert first.token_ids == [0, 1, 2, 3]
    assert manager.hash_to_block_id[first.hash] == first_id
    assert manager.blocks[tail_id].hash == -1
    assert len(manager.hash_to_block_id) == 1
    assert seq.num_cached_tokens == 0
    assert seq.num_scheduled_tokens == 7


def test_hash_blocks_respects_chunk_progress(block_module):
    manager = block_module.BlockManager(3, 4)
    seq = Sequence(list(range(9)))
    manager.allocate(seq, 0)
    seq.num_scheduled_tokens = 3
    manager.hash_blocks(seq)
    assert manager.hash_to_block_id == {}
    seq.num_cached_tokens = 3
    seq.num_scheduled_tokens = 2
    manager.hash_blocks(seq)
    assert len(manager.hash_to_block_id) == 1
    assert manager.blocks[seq.block_table[0]].token_ids == [0, 1, 2, 3]
    assert manager.blocks[seq.block_table[1]].hash == -1
    seq.num_cached_tokens = 5
    seq.num_scheduled_tokens = 3
    manager.hash_blocks(seq)
    first, second = [manager.blocks[block_id] for block_id in seq.block_table[:2]]
    assert len(manager.hash_to_block_id) == 2
    assert second.hash == manager.compute_hash([4, 5, 6, 7], first.hash)
    assert manager.blocks[seq.block_table[2]].hash == -1
    assert seq.num_cached_tokens == 5
    assert seq.num_scheduled_tokens == 3


def test_appended_token_is_not_published_before_forward(block_module):
    manager = block_module.BlockManager(1, 4)
    seq = Sequence([0, 1, 2])
    manager.allocate(seq, 0)
    seq.num_cached_tokens = 3
    seq.append_token(3)
    manager.hash_blocks(seq)
    assert manager.hash_to_block_id == {}
    seq.num_scheduled_tokens = 1
    manager.hash_blocks(seq)
    block = manager.blocks[seq.block_table[0]]
    assert block.token_ids == [0, 1, 2, 3]
    assert manager.hash_to_block_id[block.hash] == block.block_id


def test_reallocation_invalidates_overwritten_cache_entry(block_module):
    manager = block_module.BlockManager(1, 4)
    first = Sequence([0, 1, 2, 3])
    manager.allocate(first, 0)
    first.num_scheduled_tokens = 4
    manager.hash_blocks(first)
    block_id = first.block_table[0]
    previous_hash = manager.blocks[block_id].hash
    manager.deallocate(first)
    manager.allocate(Sequence([99]), 0)
    assert previous_hash not in manager.hash_to_block_id
    assert manager.blocks[block_id].hash == -1
    assert manager.blocks[block_id].token_ids == []
    assert_pool_consistent(manager)


def test_overwrite_does_not_remove_hash_mapping_to_another_block(block_module):
    manager = block_module.BlockManager(2, 4)
    first, second = Sequence([0, 1, 2, 3]), Sequence([0, 1, 2, 3])
    manager.allocate(first, 0)
    first.num_scheduled_tokens = 4
    manager.hash_blocks(first)
    manager.allocate(second, 0)
    second.num_scheduled_tokens = 4
    manager.hash_blocks(second)
    block_hash = manager.blocks[first.block_table[0]].hash
    latest_id = second.block_table[0]
    assert manager.hash_to_block_id[block_hash] == latest_id
    manager.deallocate(first)
    manager.allocate(Sequence([99]), 0)
    assert manager.hash_to_block_id[block_hash] == latest_id
    assert_pool_consistent(manager)
