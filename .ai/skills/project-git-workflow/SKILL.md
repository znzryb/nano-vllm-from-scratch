---
name: project-git-workflow
description: 在 nano-vllm-from-scratch 提交、推送或完成已授权任务需 Git 收尾时使用；仅解释代码或查看状态时不使用。
---

# 本项目提交与推送

1. 读取仓库根 `AGENTS.md` 中的「项目 Git 授权」；该节是本仓库的具体授权，源为 `.ai/rules/project-git-workflow.md`。按当前任务与提交的实际作者判断署名，不因用户写源码就把助手编写的测试也归为用户作品。
2. 检查 `git status`、staged/unstaged diff 与相关测试。保护其他任务改动；只有用户明确要求代提交时，才提交其已有学习实现，不借收尾修改实现。
3. 逐项暂存本轮文件，按实际作者决定署名，提交并按根入口授权正常推送。汇报 commit、目标分支、实际测试和未验证范围；不重复请求已有的正常 commit/push 授权。
4. 本技能与根授权均由 `.ai/` 维护，修改源后运行 `uv run python .ai/scripts/generate.py` 和 `--check`；不要直接修改 `.agents/skills/` 生成副本。
