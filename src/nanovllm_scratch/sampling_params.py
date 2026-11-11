# 对应原版 NanoVLLM 的 nanovllm/sampling_params.py。
# 职责：保存单条请求的生成参数，并在创建对象时校验参数。

# 第一步：从 dataclasses 导入 dataclass。
# 它可以根据字段声明，自动生成 __init__ 等方法。
from dataclasses import dataclass

# 第二步：定义 SamplingParams 类。
# 原版使用 dataclass 装饰器，并设置 slots=True。
# slots=True 限制实例的属性布局，避免随意添加未声明的属性。
# 使用 dataclass 后，这里不需要手写 __init__。
@dataclass(slots=True)
class SamplingParams:
    # 第三步：在类里面声明三个字段。
    # 每个字段都要写出名称、类型和默认值。
    #
    # temperature：
    # 类型为 float，默认值为 1.0。
    # 表示采样温度，后续采样模块会读取它。
    #
    # max_tokens：
    # 类型为 int，默认值为 64。
    # 表示最多新增多少个 token，不包含 prompt。
    #
    # ignore_eos：
    # 类型为 bool，默认值为 False。
    # False 表示生成 EOS 时允许结束请求。
    # True 表示忽略 EOS 停止条件，仍受最大生成长度限制。
    temperature:float = 1.0
    max_tokens:int = 64
    ignore_eos:bool = False
    def __post_init__(self):
        # 第四步：在类里面定义 __post_init__ 方法。
        # 该方法接收 self，没有其他必填参数。
        # dataclass 自动生成的 __init__ 会在字段赋值后调用它。
        # 职责：校验当前对象的参数，无业务返回值。
        #
        # 对齐原版：要求 temperature > 1e-10。
        # 不满足时明确报错，因为原版这条采样路径不支持温度为零的 greedy 模式。
        # 原版此处只检查温度，没有额外校验另外两个字段。
        # 字段上的类型注解本身不会自动执行运行时类型检查。
        if not self.temperature > 1e-10:
            raise ValueError(f"Expected temperature > 1e-10, got {self.temperature}")
        
        # 第五步：检查行为。
        # 不传参数创建对象时，三个字段应分别为 1.0、64、False。
        # 显式传入参数时，对象应保存传入的值。
        # 不同对象应能分别保存不同的生成配置。
        # temperature 为 0 或负数时，创建对象应报错。

