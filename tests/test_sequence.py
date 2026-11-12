"""Sequence 的 CPU 验收测试，对应原版 nanovllm/engine/sequence.py。

运行：uv run pytest tests/test_sequence.py -q
前七步：uv run pytest tests/test_sequence.py -q -k 'not serialization'
第八步：uv run pytest tests/test_sequence.py -q -k serialization

待实现模块：nanovllm_scratch.engine.sequence，目录名应为 engine。
延后导入使其他测试仍可收集；缺少模块或接口时如实失败，不跳过或 xfail。
生成参数使用仅含字段的测试数据，不依赖尚未建立的 SamplingParams 模块。
"""

from enum import Enum
from importlib import import_module
import pickle
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def sequence_module():
    return import_module("nanovllm_scratch.engine.sequence")


def test_status_enum_has_distinct_request_states(sequence_module):
    status = sequence_module.SequenceStatus
    assert issubclass(status, Enum)
    assert {member.name for member in status} == {"WAITING", "RUNNING", "FINISHED"}
    assert len({status.WAITING, status.RUNNING, status.FINISHED}) == 3


@pytest.mark.parametrize("prompt", [[17], [17, 3, 29]])
def test_initial_request_state(sequence_module, prompt):
    seq = sequence_module.Sequence(prompt)

    assert isinstance(seq.seq_id, int)
    assert seq.status is sequence_module.SequenceStatus.WAITING
    assert seq.is_finished is False
    assert seq.token_ids == prompt
    assert seq.token_ids is not prompt
    assert seq.last_token == prompt[-1]
    assert seq.num_tokens == len(prompt)
    assert len(seq) == len(prompt)
    assert seq.num_prompt_tokens == len(prompt)
    assert seq.num_completion_tokens == 0
    assert seq.prompt_token_ids == prompt
    assert seq.completion_token_ids == []
    assert seq.num_cached_tokens == 0
    assert seq.num_scheduled_tokens == 0
    assert seq.is_prefill is True
    assert seq.block_table == []


def test_requests_have_unique_ids(sequence_module):
    requests = [sequence_module.Sequence([17]) for _ in range(3)]
    assert len({seq.seq_id for seq in requests}) == 3


def test_constructor_copies_prompt(sequence_module):
    prompt = [17, 3, 29]
    seq = sequence_module.Sequence(prompt)
    prompt[0] = 99
    prompt.append(100)

    assert seq.token_ids == [17, 3, 29]
    assert seq.prompt_token_ids == [17, 3, 29]
    assert seq.num_tokens == seq.num_prompt_tokens == 3
    assert seq.last_token == 29


@pytest.mark.parametrize("temperature,max_tokens,ignore_eos", [
    (0.7, 9, True),
    (1.2, 1, False),
])
def test_generation_parameters_are_saved(
    sequence_module, temperature, max_tokens, ignore_eos,
):
    params = SimpleNamespace(
        temperature=temperature, max_tokens=max_tokens, ignore_eos=ignore_eos,
    )
    seq = sequence_module.Sequence([17, 3], params)

    assert seq.temperature == temperature
    assert seq.max_tokens == max_tokens
    assert seq.ignore_eos is ignore_eos


@pytest.mark.parametrize("key,expected", [
    (0, 17),
    (-1, 29),
    (slice(1, 3), [3, 29]),
    (slice(None, None, -1), [29, 3, 17]),
    (slice(3, 3), []),
])
def test_token_indexing_and_slicing(sequence_module, key, expected):
    seq = sequence_module.Sequence([17, 3, 29])
    assert seq[key] == expected


@pytest.mark.parametrize("state_name,expected", [
    ("WAITING", False), ("RUNNING", False), ("FINISHED", True),
])
def test_finished_property_tracks_status(sequence_module, state_name, expected):
    seq = sequence_module.Sequence([17])
    seq.status = getattr(sequence_module.SequenceStatus, state_name)

    assert seq.is_finished is expected


def test_append_updates_completion_and_preserves_request_metadata(sequence_module):
    prompt = [17, 3]
    seq = sequence_module.Sequence(prompt)
    seq_id = seq.seq_id
    seq.status = sequence_module.SequenceStatus.RUNNING
    seq.num_cached_tokens = 2
    seq.num_scheduled_tokens = 1
    seq.is_prefill = False
    seq.block_table = [7]

    for token, expected in [(29, [29]), (11, [29, 11])]:
        seq.append_token(token)
        assert seq.token_ids == [17, 3] + expected
        assert seq.last_token == token
        assert seq.num_tokens == len(seq) == 2 + len(expected)
        assert seq.num_prompt_tokens == 2
        assert seq.num_completion_tokens == len(expected)
        assert seq.prompt_token_ids == [17, 3]
        assert seq.completion_token_ids == expected

    assert prompt == [17, 3]
    assert seq.seq_id == seq_id
    assert seq.status is sequence_module.SequenceStatus.RUNNING
    assert seq.num_cached_tokens == 2
    assert seq.num_scheduled_tokens == 1
    assert seq.is_prefill is False
    assert seq.block_table == [7]


def test_requests_do_not_share_token_lists_or_block_tables(sequence_module):
    first = sequence_module.Sequence([17, 3])
    second = sequence_module.Sequence([17, 3])
    first.append_token(29)
    first.block_table.append(7)

    assert second.token_ids == [17, 3]
    assert second.completion_token_ids == []
    assert second.num_tokens == 2
    assert second.block_table == []


@pytest.mark.parametrize("length,expected_block_lengths", [
    (1, [1]),
    (255, [255]),
    (256, [256]),
    (257, [256, 1]),
    (512, [256, 256]),
    (513, [256, 256, 1]),
])
def test_logical_blocks_at_boundaries(sequence_module, length, expected_block_lengths):
    # 使用原版约定的 256 个 token/块；不同 token 值可暴露错误切片。
    prompt = list(range(length))
    seq = sequence_module.Sequence(prompt)
    assert seq.block_size == 256
    assert seq.num_blocks == len(expected_block_lengths)
    assert seq.last_block_num_tokens == expected_block_lengths[-1]

    blocks = [seq.block(i) for i in range(seq.num_blocks)]
    assert all(isinstance(block, list) for block in blocks)
    assert [len(block) for block in blocks] == expected_block_lengths
    assert [token for block in blocks for token in block] == prompt
    # 计算逻辑块不代表已经分配物理缓存。
    assert seq.block_table == []


@pytest.mark.parametrize("index", [-1, 2, 3])
def test_block_rejects_out_of_range_index(sequence_module, index):
    seq = sequence_module.Sequence(list(range(257)))
    # 不限制使用 assert 还是显式异常，只验证无效编号不能静默返回切片。
    with pytest.raises((AssertionError, IndexError, ValueError)):
        seq.block(index)


def test_append_crosses_full_block_boundary(sequence_module):
    seq = sequence_module.Sequence(list(range(255)))
    seq.block_table = [7]
    seq.num_cached_tokens = 255

    seq.append_token(999)
    assert seq.num_tokens == 256
    assert seq.num_blocks == 1
    assert seq.last_block_num_tokens == 256
    assert seq.block(0) == list(range(255)) + [999]

    seq.append_token(1000)
    assert seq.num_tokens == 257
    assert seq.num_blocks == 2
    assert seq.last_block_num_tokens == 1
    assert seq.block(1) == [1000]
    assert seq.num_prompt_tokens == 255
    assert seq.completion_token_ids == [999, 1000]
    assert seq.num_cached_tokens == 255
    assert seq.block_table == [7]


@pytest.mark.parametrize("is_prefill,last_state", [
    (True, [17, 3, 29, 11]),
    (False, 11),
])
def test_serialization_transfers_execution_state(sequence_module, is_prefill, last_state):
    # 第八步：使用真正的 pickle 往返，不在测试中实现状态恢复逻辑。
    seq = sequence_module.Sequence([17, 3, 29])
    seq.append_token(11)
    seq.num_cached_tokens = 2
    seq.num_scheduled_tokens = 1
    seq.block_table = [7]
    seq.is_prefill = is_prefill

    assert seq.__getstate__() == (4, 3, 2, 1, [7], last_state)
    restored = pickle.loads(pickle.dumps(seq))

    assert isinstance(restored, sequence_module.Sequence)
    assert restored.num_tokens == len(restored) == 4
    assert restored.num_prompt_tokens == 3
    assert restored.num_cached_tokens == 2
    assert restored.num_scheduled_tokens == 1
    assert restored.block_table == [7]
    assert restored.last_token == 11
    assert restored.token_ids == ([17, 3, 29, 11] if is_prefill else [])
    # 执行侧的 Decode 对象没有完整历史；不要求恢复调度状态或生成参数。
    assert seq.token_ids == [17, 3, 29, 11]


def test_serialization_can_reserialize_restored_decode_state(sequence_module):
    # 覆盖恢复后的 Decode 对象再次传输；此时 token_ids 已为空。
    seq = sequence_module.Sequence([17, 3, 29])
    seq.append_token(11)
    seq.num_cached_tokens = 2
    seq.num_scheduled_tokens = 1
    seq.block_table = [7]
    seq.is_prefill = False
    expected_state = (4, 3, 2, 1, [7], 11)

    restored = pickle.loads(pickle.dumps(seq))
    assert restored.token_ids == []
    assert restored.last_token == 11
    assert restored.is_prefill is False

    retransmitted = pickle.loads(pickle.dumps(restored))

    assert isinstance(retransmitted, sequence_module.Sequence)
    assert retransmitted.__getstate__() == expected_state
    assert len(retransmitted) == 4
    assert retransmitted.token_ids == []
    assert retransmitted.last_token == 11
    assert retransmitted.is_prefill is False
