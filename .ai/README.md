# 项目指令与技能维护

`.ai/` 是本项目唯一维护源。目前只生成 Codex 产物，不生成 `.claude/`，不修改用户级配置或参考子模块。

| 维护源 | 生成结果 | 用途 |
| --- | --- | --- |
| `INSTRUCTIONS.md` | 根目录 `AGENTS.md` | 全项目助教边界、环境、工作保护和读取约定 |
| `rules/*.md` | 适用目录的 `AGENTS.md` | 按路径条件嵌入规则正文，叠加祖先要求 |
| `skills/<name>/` | `.agents/skills/<name>/` | 原生发现的项目技能及其资源 |
| `scripts/generate.py` | `generated-manifest.json` | 记录生成物的 SHA-256 和权限 |

在仓库根目录执行：

```bash
uv run python .ai/scripts/generate.py
uv run python .ai/scripts/generate.py --check
uv run python -m unittest discover -s .ai/tests -v
```

源文件、生成器、测试和生成物一起提交，克隆后即可读取入口和发现技能。修改源后生成，再执行检查；`--check` 只检查，不写文件。不要手改生成物或生成清单。

## 规则分层

Codex 原生在启动时按仓库根到当前工作目录建立指令链；它不会因后来读取任意文件就自动应用该目录的一切规则。根入口因此明确要求：操作目标文件前，读取目标目录沿途的全部 `AGENTS.md`，叠加祖先与本层条件命中的正文。这是项目执行约定。

规则的 `paths` 是本生成器的输入格式，并非 Codex 原生自动注入功能。使用以下受限格式，双引号内容按 JSON 字符串解析；其他 YAML 语法会明确报错：

```yaml
---
paths:
  - "src/nanovllm_scratch/**"
---
```

生成器按路径的最长固定父目录放置入口，不展开当前文件清单：`src/nanovllm_scratch/**` 进入该目录，`docs/*.md` 进入 `docs/`，`**/*.cu` 留在根入口。正文保留完整仓库相对路径条件，文件不存在时也可生成相应目录入口。指向 `.ai/`、`.agents/`、`references/`、软链或独立 Git 仓库的规则留在根入口，不跨边界写入。

各链累计 UTF-8 大小以 Codex 默认 32 KiB 为上限；技能正文不嵌入指令链。规则内普通 Markdown 相对文件链接会按生成位置重定位。规则和技能中的非链接路径请明确相对于仓库根，技能内部相邻参考资源可使用相对链接。

持续生效的学习和权限边界放在入口/规则中；可复用动作流程才放进技能。短技能保持单文件，出现需要按需加载的大段细节后再拆参考文档。

## 文件保护与验收

生成前检查全部已管理文件的内容与权限；发现手改、未知原生 `AGENTS.md`/`AGENTS.override.md`、目标软链或不安全路径时停止，不覆盖或接管。被删除源文件的旧产物只有在清单管理且内容未改时才清理；未知文件始终保留。检查不遍历虚拟环境、参考目录、独立 Git 仓库和软链，也不读取凭据或运行目录。

清单用于确定所有权和检测漂移，不是防篡改签名。修改生成器时运行临时仓库行为测试，验证路由、祖先链预算、幂等、只读检查、手改保护、旧产物清理及子模块边界。原生发现还依赖客户端会话；新增或改动指令和技能后，应在新会话中确认实际加载，不把文件生成检查声称为客户端端到端验收。

## 依据

- [官方 AGENTS.md 说明](https://learn.chatgpt.com/docs/agent-configuration/agents-md)：原生指令发现、优先级和默认大小限制。
- [官方 Skills 说明](https://learn.chatgpt.com/docs/build-skills)：项目 `.agents/skills/` 发现与渐进披露。
- 项目采用 OJ AC Ladder 的 `.ai/` 单源、分层生成思路，按当前仓库范围独立实现；不引入其部署、账号和外部分支流程。
- 助教原则参考 `ai-infra-from-scratch/courses/cs336/assignment1-basics/CLAUDE.md`，并按本项目允许测试、文档及工程维护的范围改写。
