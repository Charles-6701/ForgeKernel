# 来源记录：有效区域契约检查 / Provenance: valid-region contract checks

## 机制与证据边界 / Mechanism and evidence boundary

本实验使用合成有限数据、独立 CPU 期望值、哨兵保护区和设备写入的访问计数张量，检查 `pypto.view` 有效范围与 `pypto.assemble` 回写。设备计算使用 PyPTO NPU 模式；主机分配、同步和比较使用 torch/torch_npu。

This experiment tests `pypto.view` valid extents and `pypto.assemble` writeback
with synthetic finite data, independent CPU expectations, sentinel guards and
a device-written visit tensor. Device computations use PyPTO NPU mode. Host
allocation, synchronization and comparison use torch/torch_npu.

The first proposed reusable scope is rank-2 FP32 with two partial axes and
repeated work under the configured buffer and unroll options. Each owner has
a non-overlapping valid rectangle. The counter reads an immutable seed tensor
and writes a different output tensor using ordinary PyPTO arithmetic; it is
explicitly not atomic. Scalar overlap rejection is a precondition; it
is not represented as a device measurement. An ordered duplicate launch is
the counter detector's negative control. An application must also validate
its original uninstrumented graph because instrumentation changes scheduling.

首个可复用范围是 rank-2 FP32、两个不完整轴，以及配置的缓冲/展开选项下的重复工作。每个所有者具有互不重叠的有效矩形。计数器从不可变 seed 张量读取并写入不同输出张量；它明确不是原子计数器。标量重叠拒绝是前置条件而非设备测量。有序重复 launch 是计数检测器的负向对照。插桩会改变调度，因此应用还必须验证原始未插桩图。

## Probe 状态 / Probe status

- Probe: `notes/valid_region_contract.py`.
- Previous attempt: no verdict. The driver reported a frontend
  `OperationLoopCheck: Loop Detected` build failure before any contract line.
  This is not evidence of a numerical error. The initial failure feedback
  omitted the job identifier; the later audit maps this attempt to job
  1794845022.
- 前一次尝试 / Previous attempt：无裁决。driver 在任何契约行之前报告前端 `OperationLoopCheck: Loop Detected` 构建失败；这不是数值错误证据。该次失败对应后续审计记录中的 job `1794845022`。
- Revised driver job: 1794845325, certification round 2, completed normally.
  Driver verdict: PROBE PASS max_diff=1.1920928955078125e-07.
  Final readings: `failures=[]` and
  `probe_complete=true failures=0 worst_error=1.1920928955078125e-07`.
  All contract lines passed. This completed target-NPU job is the basis for
  marking the recipe and index Verified.
- 修订后的 driver job / Revised driver job：`1794845325`，第 2 轮认证，正常完成；`PROBE PASS max_diff=1.1920928955078125e-07`，最终为 `failures=[]` 和 `probe_complete=true failures=0 worst_error=1.1920928955078125e-07`。所有契约行通过，因此 recipe 与索引可标记为 Verified。
- The probe synchronizes before observing values, guards, inputs and counts.
  Each checked contract prints exactly one `PROBE PASS/FAIL max_diff=...`
  line, including guards, values, input preservation and visits. Nonfinite
  comparisons print FAIL; build/launch exceptions cannot create a numeric PASS.
  All lines and normal job completion are required.
- The probe prints device and installed package versions, local API
  signatures/module paths, probe digest and actual scope into its driver log.
  Certification stays bound to that environment; the returned log tail does
  not repeat its earlier identity readings. No new version range is inferred.
  No timing or physical-memory claim is certified.
- probe 在观察数值、保护区、输入和计数前执行同步；每项契约恰好打印一条 `PROBE PASS/FAIL max_diff=...`。非有限比较打印 FAIL，构建/launch 异常不能产生数值 PASS。probe 还记录设备、软件包版本、本地 API 签名/路径、probe 摘要和实际范围；认证绑定于该环境，不认证计时或物理内存声明。

## 调查与重试规则 / Investigation and retry rules

Direct SSH for SDK inspection was rejected by this environment. The probe
therefore records installed local API signatures/module paths during its
driver job. No online implementation code was used. No architecture
certificate from another device tree is transferred.

A numerical/guard/count failure requires fixing or narrowing the responsible
contract before retry. A job without a verdict is a build/launch failure.
Concurrent overlapping owners require a separately tested atomic or isolated
observation mechanism. A changed runtime, layout or option needs a new driver
scope; more logical iterations cannot certify physical pipeline allocation.

数值、保护区或计数失败时，必须先修复或缩小对应契约再重试。无裁决 job 属于构建/launch 失败。并发重叠所有者需要单独测试的原子或隔离观察机制。运行时、布局或选项变化需要新的 driver 范围；增加逻辑迭代数不能认证物理流水分配。

## 构建失败后的修订 / Revision after the build failure

失败图在 tiled loop 内查看并 assemble 同一个 visit 张量。该反馈促使计数器输入输出分离，但不能证明具体是哪条编译器规则导致环，也不能证明任何公开 API 损坏。修订图只读取 seed 存储并写入独立 visit 目的张量；有序重复对照把第一次结果作为后续 launch 的输入，并使用另一个目的张量。全零 seeds 必须保持不变，正常 visits 必须为 1，重复 visits 必须为 2，且全 1 检测器必须拒绝后者。

The failed graph both viewed and assembled the same visit tensor inside the
tiled loop. That feedback motivates separating counter input from counter
output; it does not prove which compiler rule caused the cycle, or that any
public API is broken. The revised graph reads only seed storage and writes
only a distinct visit destination. The ordered duplicate control feeds the
completed first result into a later launch with another destination.
Zero seeds must stay unchanged, normal visits must be one, and duplicate
visits must be two. The all-one detector must reject the latter. Guards are
checked on every seed and visit allocation. Repeated normal launches reuse
allocations with changed synthetic data. Job 1794845325 passed every contract
line, including the duplicate rejection control, and completed with zero
failures. The complete probe's aggregate finite maximum was
1.1920928955078125e-07; that aggregate is not a per-contract error reading.

The reusable certified scope is contiguous rank-2 FP32 synthetic inputs of
83 by 131, 55 disjoint 8 by 32 logical tile owners with two partial axes,
valid-shape optimization 1, buffer option 8 and inner unroll option 4.
This establishes the observed values, guards, preserved inputs, device visit
outputs and semantic behavior under repeated allocation reuse. It does not
establish physical slot allocation or instruction overlap. No in-place or
overlapping-owner counter claim is admitted.

The reusable example and probe function are kept identical. Application
validation remains isolated to workspace notes and driver output. No
architecture, milestone or production implementation was changed.

可复用认证范围为连续 rank-2 FP32 合成输入 83×131、55 个互不重叠的 8×32 逻辑 tile 所有者、两个不完整轴、valid-shape optimization 1、缓冲选项 8 和内部展开选项 4。它只建立已观察到的数值、保护区、输入保持、设备 visit 输出和复用分配下的语义行为，不建立物理 slot 分配或指令重叠，也不接受原地或重叠所有者计数声明。可复用示例与 probe 函数保持一致；未修改架构、milestone 或生产实现。

## Agent 可读来源 / Agent-readable sources

Only public API symbols and probe verdicts support the reusable claim:
`pypto.frontend.jit`, `pypto.RunMode.NPU`, `pypto.Tensor`, `pypto.DYNAMIC`,
`pypto.DT_FP32`, `pypto.loop`, `pypto.set_vec_tile_shapes`, `pypto.view`,
`pypto.assemble`, scalar `.min`, and `torch_npu.npu.synchronize`.
Certifying verdict: PROBE PASS max_diff=1.1920928955078125e-07,
job 1794845325 (round 2); `probe_complete=true failures=0`.
Application identity, specialization and case evidence stays in the job log
and workspace notes rather than becoming reusable hardware knowledge.

中文摘要：只有公开 API 符号与 probe 裁决支持可复用声明。认证裁决为 job `1794845325`（第 2 轮）的 `PROBE PASS max_diff=1.1920928955078125e-07` 和 `probe_complete=true failures=0`。应用身份、特化和 case 证据继续保留在 job 日志与 workspace notes 中，不提升为可复用硬件知识。
