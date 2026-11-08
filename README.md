# nano-vllm-from-scratch

从零到一构建一个可理解、可验证、可测量的 LLM 推理引擎。

这个项目从空骨架开始逐步实现，参考 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)、[vLLM](https://github.com/vllm-project/vllm) 和 [SGLang](https://github.com/sgl-project/sglang) 的公开设计。它是独立学习与工程实践项目，与这些项目没有官方隶属关系。

## 目标

- 理解从请求进入到 token 输出的完整执行流程。
- 先实现正确的单请求推理，再逐步实现批处理、KV cache 和调度。
- 对每项优化保留基线、正确性验证和可复现的性能数据。
- 将验证充分、适合上游的改进整理为独立贡献。

## 当前状态

**阶段 0：项目初始化。尚未实现模型加载或推理功能，暂无性能结果。**

详细实现顺序见 [路线图](docs/roadmap.md)。不预设性能提升比例，所有结论以实测为准。

## 开发环境

使用 Python 3.12 和 uv 管理环境与依赖：

```bash
uv sync
uv run python -c "import nanovllm_scratch; print(nanovllm_scratch.__version__)"
```

推理阶段计划使用 Linux + NVIDIA CUDA GPU；环境和版本要求随实现补充。macOS 可以用于阅读、文档和不依赖 CUDA 的逻辑开发。

## 目录

```text
src/nanovllm_scratch/  推理引擎实现
docs/roadmap.md        分阶段路线与验收标准
```

## 性能实验约定

记录基线与优化 commit、GPU、模型、软件版本、精度、输入输出长度、并发和到达方式。固定请求集，预热后重复测量，报告输出 tokens/s、请求吞吐、首 token 延迟和 token 间延迟，并保留原始数据。冷启动单独测量，算子收益与端到端收益分别报告。

## License

MIT。参考或引入第三方代码时保留其许可证和来源说明。
