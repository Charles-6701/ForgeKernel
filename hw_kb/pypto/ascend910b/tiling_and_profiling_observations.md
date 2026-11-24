# 分块 launch、连续切片与 profiler 计时 / Tile launches, contiguous slices and profiling

- 平台 / Platform: Ascend 910B4（A2），CANN 9.1.0，PyPTO 0.2.1。
- 状态 / Status: 任务级实验观察。尚无独立的通用机制探针认证。
- 来源 / Sources: 复现仓库 `Charles-6701/forgekernel-ascend-repro`，
  `experiments/910b/20261123-gqa-teacher-student/`（`c9f0853`）。
  第 1 轮见 `scout-comparison.json`、`wall-comparison.json` 和 `confirmation-result.json`。
  第 2 轮约束见 `round-02/PROTOCOL.md`，作业目录为
  `/workspace/forgekernel-910b-repro/artifacts/gqa-teacher-student-r2.WYrix6`。

## 用实测 launch 数验证分块变更

当 profiler 显示分块 dispatch 开销较高时，可对尺寸满足约束的分支试验更宽的 tile。
本次实验只在 BF16、KV 长度至少 256、head dimension 不超过 128 的分支把
64-key 改为 128-key。A2 official scout 中，`gqa_4` 的 attention launch 从 4 降到 2，
`gqa_9` 从 8 降到 4。同步端到端中位延迟分别改善 2.17× 和 2.00×，同一候选
通过全部 20 个官方精度用例。性能结论限于实测形状及该计时环境；其他形状须重新验证，
尤其是资源占用、掩码、尾块和数值精度。

## 连续切片可减少动态拼装

当源地址连续、组大小整除行块且没有 padding 时，可比较按组切片与逐行切片。
本次试验只在 128-key、非因果、组大小为 4–32 且满足上述条件时启用。
第一组 A2 配对中，双方官方精度均为 3/3，`gqa_4` 和 `gqa_9` 的 launch 数一致，
去重后的任务耗时相对第 1 轮基线分别改善 1.90× 和 1.68×，控制组波动约 1%。
该证据只覆盖首组配对。接受此变更还需核对反向配对与 20 例全量精度。

## 计时和复核

同一次 PyPTO launch 的 AI CPU 与 AI Core profiler 任务区间可能重叠。对原始区间，
应按一次调用划分采样窗口，并在窗口内合并重叠区间再比较。记录 launch 数，
另用设备同步后的端到端采样复核。任务区间涵盖运行时开销，和 kernel-only 时间
属于不同口径，不能直接计算倍率。

应用本条经验后若出现退化，按 [人工审核与矫正规则](../../README.md#regression-review--退化复核)
处理。审核时优先核对实际触发的分支、shape 覆盖、launch 数和计时窗口。

## 行块打包与逻辑索引 / Row packing and logical indices

多个逻辑行共享同一个不展开的输入操作数时，可将物理连续的行一次搬入本地 tile。
源地址、掩码或其他逐行元数据、以及输出逆映射应统一用同一逻辑行标识推导。
仅在完整行组、边界和 stride 均可证明的路径使用整组搬运。用独立的行标签验证
映射，用可枚举的边界输入验证元数据扩展。连续性和逻辑等价是编译前提，
生成的指令与同步计时仍需按目标机和 shape 确认。

When several logical rows share an unexpanded input operand, load physically
contiguous rows together into a local tile. Derive source addresses, masks or
other row metadata, and inverse output mapping from the same logical row
identity. Use whole-group loads where row coverage, bounds and strides are
known. Validate the mapping with independent row tags and metadata broadcast
with enumerable boundary inputs. Check lowering and synchronized timing on
the target device and shape.

## 同步墙钟对照 / Synchronized wall-clock comparisons

以减少 launch 或缩短 profiler 任务区间为假设时，同步墙钟仍需独立验证。
在相同设备、输入、预热和采样口径下比较整个调用，计入调用内张量操作与设备
同步等待。保留原始区间、launch 数和墙钟记录。反向调整候选与基线的执行顺序，
用未改动路径估计波动。单次固定顺序比较仅覆盖其记录的形状和执行顺序。

When the hypothesis is fewer launches or shorter profiler task intervals,
validate synchronized wall time separately. Compare the complete call on
the same device with matched inputs, warmups and samples, including tensor
operations inside the call and device synchronization. Keep raw intervals,
launch counts and wall samples. Reverse the candidate/baseline execution
order and use unchanged paths to estimate drift. A fixed-order comparison
covers the recorded shape and order.

[公开数学契约与计时来源 / Public mathematical contract and timing provenance](../../../docs/kb-provenance/pypto/gqa_public_golden_and_a2_baselines.md)
