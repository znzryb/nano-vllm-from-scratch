"""RMSNorm 基础前向的验收测试，不包含待学习的算子实现。

待实现接口：nanovllm_scratch.layers.layernorm.RMSNorm(hidden_size, eps=1e-6)。
对应原版 nanovllm/layers/layernorm.py；本轮验收 forward(x)，残差融合另行测试。
输入为 [num_tokens, d_model] 或 [batch, seq, d_model]，沿最后一轴归一化。
数值参考使用已安装的 Transformers Qwen3RMSNorm，不下载权重或使用 GPU。
"""

from importlib import import_module

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


@pytest.fixture(scope="module")
def rmsnorm_class():
    # 延后导入，确保尚未实现本模块时仍可收集、运行其他模块的测试。
    # 不跳过、不 xfail：缺少实现就是当前需要看到的失败。
    return import_module("nanovllm_scratch.layers.layernorm").RMSNorm


def make_norm_pair(rmsnorm_class, d_model, *, eps=None, dtype=torch.float32):
    """两边装入相同的非均匀权重，不依赖待实现模块的初始化策略。"""
    options = {} if eps is None else {"eps": eps}
    actual = rmsnorm_class(d_model, **options).to(device="cpu", dtype=dtype)
    reference = Qwen3RMSNorm(d_model, **options).to(device="cpu", dtype=dtype)
    weight = torch.linspace(-1.5, 1.25, d_model, dtype=dtype)
    with torch.no_grad():
        actual.weight.copy_(weight)
        reference.weight.copy_(weight)
    return actual, reference


def test_weight_is_a_registered_parameter(rmsnorm_class):
    layer = rmsnorm_class(8)
    assert isinstance(layer, torch.nn.Module)
    assert isinstance(layer.weight, torch.nn.Parameter)
    assert layer.weight.shape == (8,)
    assert dict(layer.named_parameters())["weight"] is layer.weight
    assert "weight" in layer.state_dict()


@pytest.mark.parametrize("shape", [(6, 8), (1, 1, 8), (2, 3, 8), (2, 3, 1)])
@pytest.mark.parametrize("eps", [1e-6, 0.1])
def test_forward_matches_qwen3_reference(rmsnorm_class, shape, eps):
    # 每个位置的数值不同且均值不为零，可区分 RMSNorm 与 LayerNorm。
    generator = torch.Generator().manual_seed(42)
    hidden_states = torch.randn(shape, generator=generator) + 2.0
    layer, reference = make_norm_pair(rmsnorm_class, shape[-1], eps=eps)

    actual = layer(hidden_states)
    expected = reference(hidden_states)

    assert isinstance(actual, torch.Tensor)
    assert actual.shape == hidden_states.shape
    assert actual.dtype == hidden_states.dtype
    assert actual.device == hidden_states.device
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_default_epsilon_matches_qwen3(rmsnorm_class):
    layer, reference = make_norm_pair(rmsnorm_class, 4)
    hidden_states = torch.tensor([[[1e-4, -2e-4, 3e-4, -4e-4]]])
    torch.testing.assert_close(
        layer(hidden_states), reference(hidden_states), rtol=1e-5, atol=1e-7,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_zero_input_produces_finite_zeros(rmsnorm_class, dtype):
    layer, _ = make_norm_pair(rmsnorm_class, 8, dtype=dtype)
    hidden_states = torch.zeros((2, 3, 8), dtype=dtype)

    actual = layer(hidden_states)

    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, hidden_states, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [1e-4, 1000.0])
def test_low_precision_matches_reference_for_small_and_large_values(
    rmsnorm_class, dtype, scale,
):
    layer, reference = make_norm_pair(rmsnorm_class, 4, dtype=dtype)
    hidden_states = torch.tensor(
        [[[1.0, -2.0, 3.0, -4.0], [2.0, 0.0, -1.0, 0.5]]],
        dtype=dtype,
    ) * scale

    actual = layer(hidden_states)
    expected = reference(hidden_states)

    assert actual.dtype == dtype
    assert actual.shape == hidden_states.shape
    assert torch.isfinite(actual).all()
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_noncontiguous_input_matches_reference(rmsnorm_class):
    layer, reference = make_norm_pair(rmsnorm_class, 4)
    hidden_states = torch.arange(48, dtype=torch.float32).reshape(2, 4, 6)
    hidden_states = hidden_states.transpose(1, 2)
    assert hidden_states.shape == (2, 6, 4)
    assert not hidden_states.is_contiguous()

    torch.testing.assert_close(
        layer(hidden_states), reference(hidden_states), rtol=1e-5, atol=1e-6,
    )


def test_other_positions_do_not_change_a_positions_output(rmsnorm_class):
    layer, _ = make_norm_pair(rmsnorm_class, 4)
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + 1
    changed = hidden_states.clone()
    changed[0, 1:] = -100.0
    changed[1] = 1000.0

    original_output = layer(hidden_states)
    changed_output = layer(changed)

    # 修改其他 batch / seq 位置，不应影响 [0, 0] 位置。
    torch.testing.assert_close(original_output[0, 0], changed_output[0, 0])


def test_basic_forward_preserves_input_and_weight(rmsnorm_class):
    layer, _ = make_norm_pair(rmsnorm_class, 4)
    hidden_states = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]])
    original_input = hidden_states.clone()
    original_weight = layer.weight.detach().clone()

    layer(hidden_states)

    torch.testing.assert_close(hidden_states, original_input, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight, original_weight, rtol=0, atol=0)
