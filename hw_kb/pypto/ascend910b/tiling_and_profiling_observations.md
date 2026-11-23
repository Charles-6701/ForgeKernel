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
