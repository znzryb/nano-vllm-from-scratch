"""验证用户的 baseline；微型权重仅在临时目录生成，不访问 Hub。"""

import os

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


@pytest.mark.skipif(not os.environ.get("BASELINE_MODEL_PATH"),
                    reason="需要指定已下载的真实 Qwen3 权重目录")
def test_real_model_smoke():
    # 在已准备权重的 Modal 容器内也可以运行这一项；默认不使用云端 GPU。
    device = os.environ.get("BASELINE_DEVICE", "cpu")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[
        os.environ.get("BASELINE_DTYPE", "float32")
    ]
    baseline = BaseLine(os.environ["BASELINE_MODEL_PATH"], device=device, dtype=dtype)
    input_ids = baseline.tokenizer("Hello", return_tensors="pt")["input_ids"]
    logits = baseline.forward(input_ids)
    assert logits.shape == (1, input_ids.shape[1], baseline.config.vocab_size)
    assert logits.device.type == torch.device(device).type
    assert torch.isfinite(logits).all()
    assert not logits.requires_grad
