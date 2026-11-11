---
paths:
  - "**"
---

# 项目 Git 授权

- 本项目的 Git commit message（标题与正文）、PR 标题和正文统一使用中文；代码标识符、命令、路径、专有名词及 `Co-Authored-By` 等标准 Git trailer 保留原文。
- 本仓库 `znzryb/nano-vllm-from-scratch` 是用户独立学习项目，`origin` 为 `git@github.com:znzryb/nano-vllm-from-scratch.git`。参考 NanoVLLM 或引入子模块不使本仓库的提交自动成为对上游的贡献；今后实际向别人上游提 PR 时另按对外贡献判断。
- 用户持续授权：完成已授权任务后，可以自动提交该任务实际产生的文件并正常推送到本仓库 `origin`。当前使用 `main`；如项目以后建立分支/PR 流程，按新流程推送对应分支，不擅自合并。无需逐次询问 commit 和普通 push。
- 自动提交仅限当前任务成果；用户或其他会话已有改动必须保留，只有用户明确要求代提交时才能纳入。Git 收尾授权不授予修改用户源码的权限。
- 本仓库中，AI 实质编写的提交自动添加实际助手的 Co-Authored-By；Codex 使用 `Codex <noreply@openai.com>`。用户主要编写、助手仅解释/review/代提交的内容不添加 AI 署名；混合内容尽可能分开提交。向他人上游贡献时默认不加 AI 署名，且遵守上游披露要求。
- 提交前检查 staged/unstaged diff 与相关测试，只逐项列具体路径暂存；不更改 Git 身份，不带入凭据、模型权重、缓存或无关文件。强推、历史改写、远端删除及 PR 合并不在本授权内。
