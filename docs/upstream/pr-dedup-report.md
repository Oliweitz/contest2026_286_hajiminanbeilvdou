# 上游查重报告（2026-09-17）

提 PR 前的撞车检查：把我们的修复与三个上游仓**当前 `dev-ai-contest-2026` 分支**逐一比对。

## 结论速览

| 结论 | 数量 | 说明 |
|---|---|---|
| ✅ **未修复，PR 有独特价值** | 20+ | 我们的大部分修复上游都没有 |
| ⚠️ **部分重叠** | 1 | `vela_tls.c`：上游 PR#27 修了「读」，我们修的是「写」，**互补不冲突** |
| 🔍 **未逐一细看** | 少量 | 仅核对了文件级历史与关键词，未逐行 diff |

**三个补丁在上游当前分支上 `git apply --check` 全部干净通过**，
可直接建分支提 PR，不需要 rebase。

## 一、packages_ai_agent（补丁 01，26 个文件）

检查方式：文件级提交历史 + 关键代码直查（上游共有 38 个 PR 引用，抽查了最近 25 个的标题与 diff 范围）。

| # | 我们的修复 | 上游状态 | 证据 |
|---|---|---|---|
| A20 | `vela_tls` 写请求 >16 KB 必失败 | **未修** | 上游 `tls_write_request` 仍把 `body_len` 整包传给 `mbedtls_ssl_write`（`vela_tls.c:477-495`） |
| A3 | 缓存命中跳过工具循环 | **未修** | `agent_loop.c` 自 2026-06-02 后无人改，缓存逻辑无 `used_tools` 概念 |
| A5 | `REST_API=y` 时编译不过 | **未修** | `api_handler.c:535` 仍调 `agent_logbuf_dump()` 却无 include；`llm_port` 映射仍缺 |
| A21 | 时钟校验只查下界 | **未修** | `tool_get_time.c:167` 仍是 `if (now > 1735689600)` |
| A22 | `%*[^,]` 扫描集 NuttX 不支持 | **未修** | `tool_get_time.c:47` 原样 |
| A23 | 技能描述死代码（12 个技能全失效） | **未修** | `skill_loader.c:289` 起 `extract_description` 原样，该文件自 initial commit 后无人改 |
| A1/A6/A16/A17/A19 | SO_LINGER / 工具白名单 / WS 无锁写 / Nagle / logbuf | **未修** | 对应文件无相关提交；`ws_server.c` 仅有的上游改动是删死函数与 REST 端点新增 |
| A7 | `lvgl_ui_channel` 编译与宿主假设 | **未修** | `lvgl_ui_channel.c` 自 initial commit 后无人改 |
| A2 | Kconfig 缺 `NET_TCPBACKLOG` select | **未修** | 上游 Kconfig 无任何 select；参考 defconfig `gemini-s1` 仍写 `CONFIG_NET_TCPBACKLOG=y` |
| A4 | heartbeat 退出崩溃 | **未修（未列在补丁内）** | — |

**唯一重叠**：

- **PR#27**（2026-08-18，`fix: stop chunked response read at terminator block`）——
  只改 `vela_tls.c` 的 **`tls_read_response`**：chunked 响应读到终止块就停，避免 keep-alive 等 60 秒触发 LLM 看门狗。
  我们的 A20 修的是 **`tls_write_request`**（请求侧单次写上限于 16 KB）。
  **不同函数、不同缺陷**，两处都需要。我们的补丁已包含读侧的独立性（应用后与 PR#27 共存无冲突）。

## 二、vendor_sifli（补丁 02，8 个文件）

| 我们的修复 | 上游状态 |
|---|---|
| 新增 `sf32lb_audcodec.c/.h`（全树首个可用录音驱动） | **没有同类实现**（`chips/sf32lb52/` 无 audio 文件） |
| 板级 `configs/agent/` | **不存在**（只有 nsh 等） |
| GCC14 缺原型三处（A15） | **未修** |
| AUDCODEC HAL 五处缺陷（A8–A12） | **未修**（HAL 文件无人动） |

## 三、nuttx-apps（补丁 03，6 个文件）

| 我们的修复 | 上游状态 | 证据 |
|---|---|---|
| A13 pppd 写死拨号脚本 + `/dev/ttyS1` | **未修** | `pppd_main.c:53-79` 仍是 `ATE1/ATD*99#` + `/dev/ttyS1` |
| A14 缺 `local_ip` 字段 | **未修** | `pppd.h` 无该字段 |
| A18 直连场景误判离线自毁 | **未修** | `ppp_conf.h:61` 仍是 `AHDLC_TX_OFFLINE 5`；`ppp.c:221` 仍是每次 poll 递增的 `ip_no_data_time` |

## 四、提 PR 时的注意事项

1. **三个补丁在上游当前分支上应用干净**——无需 rebase，直接建分支即可。
2. **`packages_ai_agent` 的补丁内容较杂**（21 个混合提交：修复 + 新功能）。
   PR 描述里建议按「缺陷修复」与「新功能」分节，方便 reviewer（见 `pr-guide.md`）。
3. **可顺带在 PR 描述中提及**：A2 的 Kconfig `select`、A4 的退出崩溃、A8–A12 的 HAL 五处
   ——这些不在本补丁内但同一批发现，供组委会归并处理。
4. 上游 `dev` 分支比 `dev-ai-contest-2026` 更新（如 `packages_ai_agent/dev` 有火山引擎 ASR）。
   本报告与 PR 均针对**赛事分支** `dev-ai-contest-2026`（比赛规定的目标分支）。
