"""验证用户的 baseline；微型权重仅在临时目录生成，不访问 Hub。"""

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from nanovllm_scratch.baseline import BaseLine


@pytest.fixture(scope="module")
def tiny_baseline(tmp_path_factory):
    # 使用第三方框架现成模型作为测试数据，不实现另一套前向。
    directory = tmp_path_factory.mktemp("tiny-qwen3")
    config = Qwen3Config(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=8,
        attention_dropout=0.1,
        pad_token_id=0,
        eos_token_id=2,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reference = Qwen3ForCausalLM(config).eval()
    reference.save_pretrained(directory)

    vocabulary = {word: i for i, word in enumerate(
        ["[PAD]", "[UNK]", "[EOS]", "hello", "world", "a", "b", "c"]
    )}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(directory)
    baseline = BaseLine(str(directory), device="cpu", dtype=torch.float32)
    return baseline, reference


def test_initialization_and_forward_match_reference(tiny_baseline):
    baseline, reference = tiny_baseline
    assert baseline.max_sequence_len == 8
    assert baseline.eos_token_id == 2
    assert not baseline.model.training
    assert not baseline.config.use_cache
    assert all(not p.requires_grad for p in baseline.model.parameters())

    input_ids = baseline.tokenizer("hello world", return_tensors="pt")["input_ids"]
    actual = baseline.forward(input_ids)
    with torch.inference_mode():
        expected = reference(input_ids, use_cache=False, return_dict=True).logits
    assert actual.shape == (1, 2, 8)
    assert actual.device.type == "cpu"
    assert torch.isfinite(actual).all()
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("length", [1, 8])
def test_accepts_sequence_length_boundaries(tiny_baseline, length):
    baseline, _ = tiny_baseline
    # 同时包含 token ID 的合法下界 0 和上界 vocab_size - 1。
    input_ids = torch.tensor([[i % 2 * 7 for i in range(length)]])
    assert baseline.forward(input_ids).shape == (1, length, 8)


@pytest.mark.parametrize("value, error", [
    ([1, 2], TypeError),
    (torch.tensor([[1.0]]), TypeError),
    (torch.tensor(1), ValueError),
    (torch.tensor([1, 2]), ValueError),
    (torch.ones((1, 1, 1), dtype=torch.long), ValueError),
    (torch.tensor([[1], [2]]), ValueError),
    (torch.empty((0, 1), dtype=torch.long), ValueError),
    (torch.empty((1, 0), dtype=torch.long), ValueError),
    (torch.ones((1, 9), dtype=torch.long), ValueError),
    (torch.tensor([[-1]]), ValueError),
    (torch.tensor([[8]]), ValueError),
])
def test_rejects_invalid_input(tiny_baseline, value, error):
    baseline, _ = tiny_baseline
    with pytest.raises(error):
        baseline.forward(value)


def test_future_tokens_do_not_change_prefix_logits(tiny_baseline):
    baseline, _ = tiny_baseline
    first = baseline.forward(torch.tensor([[3, 4, 5]]))
    second = baseline.forward(torch.tensor([[3, 4, 6]]))
    torch.testing.assert_close(first[:, :2], second[:, :2])


def test_repeated_calls_do_not_leak_request_state(tiny_baseline):
    baseline, _ = tiny_baseline
    input_ids = torch.tensor([[3, 4]])
    first = baseline.forward(input_ids)
    baseline.forward(torch.tensor([[5, 6, 7]]))
    torch.testing.assert_close(baseline.forward(input_ids), first)


def scripted_generator(frames=(), eos_token_id=2):
    """固定前向输出以隔离生成控制流；不实现模型或另一套解码算法。"""
    baseline = BaseLine.__new__(BaseLine)
    baseline.device = "cpu"
    baseline.config = SimpleNamespace(vocab_size=8)
    baseline.max_sequence_len = 8
    baseline.eos_token_id = eos_token_id
    baseline.forward = Mock(side_effect=list(frames))
    return baseline


def test_generate_zero_tokens_does_not_call_forward():
    baseline = scripted_generator()
    assert baseline.generate(torch.tensor([[3, 4]]), max_new_tokens=0) == []
    baseline.forward.assert_not_called()


@pytest.mark.parametrize("distract_with_earlier_position", [False, True])
def test_generate_selects_token_from_last_position(distract_with_earlier_position):
    logits = torch.full((1, 2, 8), -10.0)
    logits[0, -1, 3] = 10.0
    if distract_with_earlier_position:
        logits[0, 0, 5] = 100.0
    baseline = scripted_generator([logits])
    assert baseline.generate(torch.tensor([[6, 7]]), max_new_tokens=1) == [3]


def test_generate_includes_first_eos_and_stops_immediately():
    logits = torch.zeros((1, 1, 8))
    logits[0, 0, 2] = 10.0
    baseline = scripted_generator([logits])
    assert baseline.generate(torch.tensor([[3]]), max_new_tokens=4) == [2]
    assert baseline.forward.call_count == 1


def test_generate_stops_at_eos_after_a_regular_token():
    first = torch.zeros((1, 1, 8))  # 相同分数时，greedy 选择最小索引 0。
    second = torch.zeros((1, 2, 8))
    second[0, -1, 2] = 10.0
    # 额外输出使错误实现继续执行后也能报告结果差异，而非耗尽 Mock。
    baseline = scripted_generator([
        first, second, torch.zeros((1, 3, 8)), torch.zeros((1, 4, 8)),
    ])
    assert baseline.generate(torch.tensor([[3]]), max_new_tokens=4) == [0, 2]
    assert baseline.forward.call_count == 2


@pytest.mark.parametrize("eos_token_id", [2, None])
def test_generate_passes_complete_sequence_and_preserves_prompt(eos_token_id):
    frames = [torch.zeros((1, length, 8)) for length in (2, 3, 4)]
    baseline = scripted_generator(frames, eos_token_id=eos_token_id)
    prompt = torch.tensor([[3, 2]])  # prompt 结尾的 EOS 不应阻止生成。
    original = prompt.clone()
    result = baseline.generate(prompt, max_new_tokens=3)
    assert result == [0, 0, 0]
    assert all(type(token) is int for token in result)
    assert baseline.forward.call_count == 3
    for call, expected in zip(baseline.forward.call_args_list,
                              ([[3, 2]], [[3, 2, 0]], [[3, 2, 0, 0]])):
        actual = call.args[0]
        assert actual.dtype == torch.long
        assert actual.device.type == "cpu"
        torch.testing.assert_close(actual, torch.tensor(expected))
    torch.testing.assert_close(prompt, original)


@pytest.mark.parametrize("value, error", [
    (True, TypeError), (False, TypeError), (1.5, TypeError),
    ("3", TypeError), (None, TypeError), (-1, ValueError),
])
def test_generate_rejects_invalid_token_budget(value, error):
    baseline = scripted_generator()
    with pytest.raises(error):
        baseline.generate(torch.tensor([[3]]), max_new_tokens=value)
    baseline.forward.assert_not_called()


@pytest.mark.parametrize("value, error", [
    ([1, 2], TypeError),
    (torch.tensor([[1.0]]), TypeError),
    (torch.tensor(1), ValueError),
    (torch.tensor([1, 2]), ValueError),
    (torch.ones((1, 1, 1), dtype=torch.long), ValueError),
    (torch.tensor([[1], [2]]), ValueError),
    (torch.empty((0, 1), dtype=torch.long), ValueError),
    (torch.empty((1, 0), dtype=torch.long), ValueError),
    (torch.ones((1, 9), dtype=torch.long), ValueError),
    (torch.tensor([[-1]]), ValueError),
    (torch.tensor([[8]]), ValueError),
])
def test_generate_validates_prompt_even_with_zero_budget(value, error):
    baseline = scripted_generator()
    with pytest.raises(error):
        baseline.generate(value, max_new_tokens=0)
    baseline.forward.assert_not_called()


def test_generate_accepts_exact_context_budget():
    baseline = scripted_generator([torch.zeros((1, 2, 8))], eos_token_id=0)
    assert baseline.generate(torch.tensor([[3, 4]]), max_new_tokens=6) == [0]


def test_generate_rejects_context_overflow_before_forward():
    baseline = scripted_generator()
    with pytest.raises(ValueError):
        baseline.generate(torch.tensor([[3, 4]]), max_new_tokens=7)
    baseline.forward.assert_not_called()


def test_generate_keeps_requests_independent():
    baseline = scripted_generator([
        torch.zeros((1, length, 8)) for length in (1, 2, 1, 2)
    ], eos_token_id=None)
    first = baseline.generate(torch.tensor([[3]]), max_new_tokens=2)
    second = baseline.generate(torch.tensor([[4]]), max_new_tokens=2)
    assert first == second == [0, 0]
    assert first is not second
    torch.testing.assert_close(baseline.forward.call_args_list[2].args[0],
                               torch.tensor([[4]]))


def test_generate_matches_transformers_greedy_on_tiny_model(tiny_baseline):
    baseline, reference = tiny_baseline
    prompt = torch.tensor([[3, 4]])
    with torch.inference_mode():
        expected = reference.generate(
            prompt,
            max_new_tokens=3,
            do_sample=False,
            use_cache=False,
            eos_token_id=2,
            pad_token_id=0,
        )
    assert baseline.generate(prompt, max_new_tokens=3) == expected[0, 2:].tolist()


def run_example(*arguments):
    """通过公开命令行入口验证示例，不提供待实现的文本生成适配层。"""
    return subprocess.run(
        [sys.executable, "-m", "nanovllm_scratch.example", *arguments],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )


def test_example_help_without_loading_a_model():
    result = run_example("--help")
    assert result.returncode == 0, result.stderr
    for option in ("--model-path", "--prompt", "--max-new-tokens", "--device", "--dtype"):
        assert option in result.stdout


@pytest.mark.parametrize("max_new_tokens", [0, 3])
def test_example_prints_only_decoded_new_tokens(tiny_baseline, max_new_tokens):
    """stdout 仅输出新增文本；跳过特殊 token，允许末尾一个换行。"""
    baseline, reference = tiny_baseline
    prompt = "hello world"
    input_ids = baseline.tokenizer(prompt, return_tensors="pt")["input_ids"]
    if max_new_tokens == 0:
        expected_text = ""
    else:
        with torch.inference_mode():
            expected_ids = reference.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=False,
                eos_token_id=2, pad_token_id=0,
            )
        expected_text = baseline.tokenizer.decode(
            expected_ids[0, input_ids.shape[1]:], skip_special_tokens=True,
        )
    result = run_example(
        "--model-path", baseline.tokenizer.name_or_path,
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--device", "cpu", "--dtype", "float32",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout in (expected_text, expected_text + "\n")


def test_example_reports_invalid_token_budget(tiny_baseline):
    baseline, _ = tiny_baseline
    result = run_example(
        "--model-path", baseline.tokenizer.name_or_path,
        "--prompt", "hello world", "--max-new-tokens", "-1",
        "--device", "cpu", "--dtype", "float32",
    )
    assert result.returncode != 0
    assert "max_new_tokens" in result.stderr or "--max-new-tokens" in result.stderr
    assert result.stdout == ""


@pytest.mark.skipif(not os.environ.get("BASELINE_MODEL_PATH"),
                    reason="需要指定已下载的真实 Qwen3 权重目录")
def test_real_model_smoke():
    # 在已准备权重的 Modal 容器内也可以运行这一项；默认不使用云端 GPU。
    device = os.environ.get("BASELINE_DEVICE", "cpu")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[
        os.environ.get("BASELINE_DTYPE", "float32")
    ]
    baseline = BaseLine(os.environ["BASELINE_MODEL_PATH"], device=device, dtype=dtype)
    prompt = "The capital of France is"
    input_ids = baseline.tokenizer(prompt, return_tensors="pt")["input_ids"]
    logits = baseline.forward(input_ids)
    assert logits.shape == (1, input_ids.shape[1], baseline.config.vocab_size)
    assert logits.device.type == torch.device(device).type
    assert torch.isfinite(logits).all()
    assert not logits.requires_grad

    max_new_tokens = 16
    new_token_ids = baseline.generate(input_ids, max_new_tokens=max_new_tokens)
    assert 0 < len(new_token_ids) <= max_new_tokens
    assert all(type(token) is int and 0 <= token < baseline.config.vocab_size
               for token in new_token_ids)
    generated_text = baseline.tokenizer.decode(new_token_ids, skip_special_tokens=True)
    # 使用 pytest -s 显示真实模型的续写；不对自然语言内容作固定断言。
    print(f"\n设备：{device}，精度：{dtype}")
    print(f"输入：{prompt}")
    print(f"新增 token IDs：{new_token_ids}")
    print(f"生成文本：{generated_text}")
