"""SamplingParams 的 CPU 验收测试，对应原版 nanovllm/sampling_params.py。

覆盖默认值、构造参数、实例隔离、slots 和温度校验。
只约束原版已有行为，不额外要求 max_tokens 或 ignore_eos 的运行时校验。
"""

from importlib import import_module

import pytest


@pytest.fixture(scope="module")
def sampling_params_class():
    # 延后导入，未完成的学习实现不会阻止其他模块收集测试。
    return import_module("nanovllm_scratch.sampling_params").SamplingParams


def test_default_parameters(sampling_params_class):
    params = sampling_params_class()
    assert params.temperature == 1.0
    assert params.max_tokens == 64
    assert params.ignore_eos is False


@pytest.mark.parametrize("temperature,max_tokens,ignore_eos", [
    (0.7, 9, True),
    (1.2, 1, False),
])
def test_constructor_accepts_custom_parameters(
    sampling_params_class, temperature, max_tokens, ignore_eos,
):
    params = sampling_params_class(
        temperature=temperature, max_tokens=max_tokens, ignore_eos=ignore_eos,
    )
    assert params.temperature == temperature
    assert params.max_tokens == max_tokens
    assert params.ignore_eos is ignore_eos


@pytest.mark.parametrize("overrides,expected", [
    ({"temperature": 0.5}, (0.5, 64, False)),
    ({"max_tokens": 8}, (1.0, 8, False)),
    ({"ignore_eos": True}, (1.0, 64, True)),
])
def test_unspecified_parameters_keep_defaults(sampling_params_class, overrides, expected):
    params = sampling_params_class(**overrides)
    assert (params.temperature, params.max_tokens, params.ignore_eos) == expected


@pytest.mark.parametrize("temperature", [-1.0, 0.0, 5e-11, 1e-10])
def test_constructor_rejects_temperature_at_or_below_threshold(
    sampling_params_class, temperature,
):
    # 必须在构造时校验，不能依赖调用者手动调用 __post_init__。
    # TypeError 不算通过，避免把“不接受构造参数”误判成温度校验成功。
    with pytest.raises((AssertionError, ValueError)):
        sampling_params_class(temperature=temperature)


@pytest.mark.parametrize("temperature", [2e-10, 0.7, 2.0])
def test_constructor_accepts_temperature_above_threshold(
    sampling_params_class, temperature,
):
    params = sampling_params_class(temperature=temperature)
    assert params.temperature == temperature


def test_changing_one_instance_does_not_change_another(sampling_params_class):
    first = sampling_params_class()
    second = sampling_params_class()
    first.temperature = 0.5
    first.max_tokens = 8
    first.ignore_eos = True

    assert (first.temperature, first.max_tokens, first.ignore_eos) == (0.5, 8, True)
    assert (second.temperature, second.max_tokens, second.ignore_eos) == (1.0, 64, False)
    fresh = sampling_params_class()
    assert (fresh.temperature, fresh.max_tokens, fresh.ignore_eos) == (1.0, 64, False)


def test_slots_rejects_undeclared_attributes(sampling_params_class):
    params = sampling_params_class()
    # 通过外部行为检查 slots，例如参数名拼错时不能静默新建属性。
    with pytest.raises(AttributeError):
        params.temprature = 0.5
