# Ascend 910B 上的 PyPTO / PyPTO on Ascend 910B

这是 Ascend 910B 的专属知识库树。它有意从空目录开始：其他架构目录中的 recipe 不能作为本架构的证据。

This is the exact knowledge-base tree for Ascend 910B. It starts empty on purpose:
no recipe in another architecture's tree is evidence about this one.

## 本地接口 / Local interfaces

- Use the task's frozen PyPTO/PTO SDK and the official CANN-Bench adapter.
- Start from `../TEMPLATE.md` when adding a recipe.
- Confirm the installed PyPTO version and platform configuration before relying on
  API details from a different Ascend variant.

中文要求：使用任务冻结的 PyPTO/PTO SDK 和官方 CANN-Bench adapter；新增 recipe 时从 `../TEMPLATE.md` 开始；引用其他 Ascend 变体的 API 细节前，必须确认本机 PyPTO 版本和平台配置。

## 测量 / Measurement

Use the card's declared `timing_basis`; do not mix `cann_trace_view`,
`cann_kernel_details` or msprof task rows. A probe must run on the target
hardware before a recipe is marked Verified.

使用任务卡声明的 `timing_basis`，不得混用 `cann_trace_view`、`cann_kernel_details` 或 msprof task 行。recipe 标记为 Verified 前，probe 必须在目标硬件上完成。

## 已验证 recipe / Verified recipes

| Recipe / 配方 | Status and certification / 状态与认证 | Probe / 探针 | Scope / 范围 |
| --- | --- | --- | --- |
| [带保护区的有效区域回写与互斥所有者访问计数 / Guarded valid-region writeback and disjoint-owner visit counters](valid_region_contract.md) | 已验证 / Verified — `PROBE PASS max_diff=1.1920928955078125e-07`，job `1794845325`（第 2 轮 / round 2） | `notes/valid_region_contract.py` | 合成连续 rank-2 FP32、不完整有效范围、互斥非原子 visits、保护区、输入保持和重复分配语义；不声明物理分配或速度。 / Synthetic contiguous rank-2 FP32, partial valid extents, disjoint non-atomic visits, guards, input preservation and repeated-allocation semantics; no physical allocation or speed claim. |

## 实验观察 / Experimental observations

- [分块 launch、连续切片与 profiler 计时](tiling_and_profiling_observations.md)
  记录 A2 的实测范围和待验证项。此类观察不能直接作为通用 Verified recipe 使用。
- 应用条目后若出现退化，遵循 [人工审核与矫正规则](../../README.md#regression-review--退化复核)。

## 当前缺口 / Current gaps

Physical UB/L1 allocation, overlap, stalls and performance are outside this
certificate. Do not copy a 950PR certification into this tree merely because
the API looks similar.

物理 UB/L1 分配、执行重叠、停顿和性能均不在本证书范围内。即使 API 外观相似，也不得把 950PR 的认证复制到本目录。
