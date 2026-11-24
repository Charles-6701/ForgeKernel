# 公开 GQA golden 与 A2 基线 / Public GQA golden and A2 baselines

## 公开数学契约 / Public mathematical contract

本地公开任务说明包含 golden 的源码摘录。A2 计时脚本直接执行 CANN-Bench
`059cb61d41b279ad021160517bd2a5f6b7a05e01` 的
`tasks/level4/gqa/golden.py::gqa`，归档记录了 golden 和用例表的 SHA-256。
下面的公式来自公开任务说明，性能仅适用于计时归档中的 A2 环境。
[公开任务说明](../../../tasks/gqa-pypto/description.md) ·
[A2 基线及源码哈希](https://github.com/Charles-6701/forgekernel-ascend-repro/blob/702b4fc7c9e13e1b8a7d5f16a63ee8d03d412187/experiments/910b/20261123-gqa-teacher-student/pytorch-reference/result.json)

The local public task description includes the golden source excerpt. On A2,
the benchmark called `tasks/level4/gqa/golden.py::gqa` directly from the
pinned CANN-Bench checkout above and recorded the golden and case-table hashes.
The formulas below come from the public task description. The measurements
apply to the archived A2 configuration.
[Public task description](../../../tasks/gqa-pypto/description.md) ·
[A2 baseline and source hashes](https://github.com/Charles-6701/forgekernel-ascend-repro/blob/702b4fc7c9e13e1b8a7d5f16a63ee8d03d412187/experiments/910b/20261123-gqa-teacher-student/pytorch-reference/result.json)

| 项目 / Item | 契约 / Contract |
| --- | --- |
| 分组 / Grouping | 令 `G=Nq/Nkv`，query head `h` 使用 KV head `floor(h/G)`。Golden 将组内 head 与 query 序列折成矩阵乘法的行，复用原有 KV heads。 / Query head `h` uses KV head `floor(h/G)`. The golden folds the within-group head and query position into the matmul row dimension and reuses the original KV heads. |
| 缩放 / Scale | Golden 对所有有限 `scaleValue <= 0` 使用 `1/sqrt(D)`，正值直接使用。20 个官方用例只覆盖默认 `-1.0` 和显式 `0.08838`。R4 在代码中仅接受 `-1.0` 或正有限值，官方通过不覆盖其他非正值。 / The golden uses `1/sqrt(D)` for finite nonpositive scales and uses a positive scale as passed. The 20 official cases cover only `-1.0` and `0.08838`. R4 accepts `-1.0` or positive finite values, leaving other nonpositive scales outside that pass. |
| 因果 / Causality | 右下角对齐，位置 `i` 可见且仅可见 `j <= i + Sk - Sq` 的 key。Golden 代码把全屏蔽行的权重置零；公开因果用例满足 `Sq <= Sk`。 / Bottom-right aligned. Query position `i` sees exactly keys `j <= i + Sk - Sq`. The golden explicitly zeros the weights of fully masked rows. Public causal cases satisfy `Sq <= Sk`. |
| 数值 / Numerics | Golden 分数经缩放和掩码后，在 key 轴上 softmax，再与 V 相乘。PyPTO 的流式 FP32 最大值、分母和分子可经过每 tile 的共同缩放计算同一表达式。概率转换至公开输入 dtype 会引入舍入，仍需官方 checker。 / The golden normalizes masked, scaled scores over keys before multiplying by V. Streaming FP32 maxima, denominators and numerators represent the same expression when both sums are rescaled consistently at each tile. Casting probabilities to the public input dtype introduces rounding, so the official checker remains required. |

Golden 折叠后的行轴是组内 head 优先，现有 PyPTO owner 使用 query 位置优先。
两种顺序可独立计算同一组行，前提是 q、因果上界和输出逆映射始终匹配同一个
`(batch, KV head, position, within-group head)`。独立探针可为每个 KV head 和
position 加不同标签。对因果路径，使用 Q=K=0 和可独立枚举的 V，期望每行输出为
所有可见 V 的均值。改变最后一块的 V 可检查后半段 key 是否实际参与。

The folded golden orders within-group heads before positions. The PyPTO owner
orders positions before heads. Either row order yields the same mathematical
rows when the Q load, causal limit and final store consistently refer to the
same `(batch, KV head, position, within-group head)`. Tag distinct KV heads
and positions in an independent probe. With Q=K=0, each causal output row
should equal the mean of independently enumerated visible V values.
Changing only the last tile's V checks that the late keys contribute.

## A2 测量边界 / A2 measurement scope

- **公开 PyTorch eager golden。** Ascend 910B4、CANN 9.1.0、TorchNPU
  2.10.0.post4。固定种子，预热 2 次，取 5 次设备同步墙钟样本的中位数。
  20 个延迟中位数的几何平均为 **1.4944 ms**。`gqa_8` 为 **8.1898 ms**。
  / **Public PyTorch eager golden.** Fixed-seed inputs, two warmups and median
  of five device-synchronized wall-clock samples on the same recorded stack.
  Geometric mean of 20 medians **1.4944 ms**, with `gqa_8` at **8.1898 ms**.
  [计时入口及完整结果 / Benchmark and complete results](https://github.com/Charles-6701/forgekernel-ascend-repro/blob/702b4fc7c9e13e1b8a7d5f16a63ee8d03d412187/experiments/910b/20261123-gqa-teacher-student/pytorch-reference/RESULT.md)
- **R4 PyPTO 对照。** 固定顺序在 eager 之后补测同 shape、dtype、数据范围和种子的
  `gqa_8`，同步墙钟为 **2861.6052 ms**，本次 PyPTO/eager 延迟比为
  **349.41×**。这一 PyPTO/eager 对照只覆盖该用例。独立官方全量记录证明 R4
  **20/20** 正确，R4 对 R2 的 `gqa_8` 正反配对提速几何平均为 **1.845×**，
  attention launch **16→8**，归一化仍为 1。
  / **R4 PyPTO comparison.** A fixed-order follow-up using the same
  `gqa_8` shape, dtype, input range and seed yielded **2861.6052 ms**,
  or **349.41×** the eager latency for this case. The separate official
  evaluation verified R4 at **20/20**. The two-order R4/R2 comparison
  recorded **1.845×** geometric-mean speedup on `gqa_8`, with
  attention launches **16→8** and one normalization launch.
  [Eager 与 PyPTO 对照 / Eager comparison](https://github.com/Charles-6701/forgekernel-ascend-repro/blob/702b4fc7c9e13e1b8a7d5f16a63ee8d03d412187/experiments/910b/20261123-gqa-teacher-student/pytorch-reference/RESULT.md) ·
  [R4 正反配对与全量确认 / R4 pairing and full confirmation](https://github.com/Charles-6701/forgekernel-ascend-repro/blob/702b4fc7c9e13e1b8a7d5f16a63ee8d03d412187/experiments/910b/20261123-gqa-teacher-student/round-04/RESULT.md)
- **独立库参考。** A2 CANN-Bench 封存库参考采用 `cann_kernel_details` 的
  kernel-only 窗口，20 例几何平均 **0.184798 ms**。该数值和 eager 同步墙钟
  具有不同测量边界，分别记录。库内部实现未参与本分析。
  / **Separate library reference.** The sealed library reference checked
  with `cann_kernel_details` reports a **0.184798 ms** 20-case
  geometric mean for its kernel-only window. Record this separately from
  eager synchronized wall time. Library internals are outside this analysis.
  [A2 库参考记录 / A2 library reference record](https://github.com/Charles-6701/forgekernel-ascend-repro/blob/702b4fc7c9e13e1b8a7d5f16a63ee8d03d412187/experiments/910b/20261123-gqa-teacher-student/round-01/RESULT.md)

R6 在短 query 上将 attention launch 减半，去重 profiler 区间明显缩短，
同步墙钟几何平均仅改善 **1.021×**，低于对照波动 **1.031×** 和既定门槛 **1.10×**。
这支持分开记录调用级墙钟、profiler 任务区间和 launch 数。下一轮优先测试因果
长序列 owner 内的连续 Q head 组搬运，并验证组内重复的因果上界。只改变
packing，保留 R4 的 tile、精度和 launch 数。先做独立因果边界探针，再做官方
scout 与正反同步计时，20 例全量通过后方可晋升。

On short queries, R6 halved attention launches and reduced the
de-overlapped profiler intervals, while the synchronized geometric-mean gain
of **1.021×** was below both **1.031×** control drift and the **1.10×**
gate. Keep wall time, profiler task intervals and launch counts separate.
The next proposed experiment packs contiguous Q head groups in long causal
owners and verifies that each group repeats its position's causal limit.
It retains R4's tile, precision and launch count. Check the independent
causal boundary, official scout and both timing orders, then run the 20-case
gate before promoting the candidate.
[R6 配对与晋升判定 / R6 pairing and gate](https://github.com/Charles-6701/forgekernel-ascend-repro/blob/702b4fc7c9e13e1b8a7d5f16a63ee8d03d412187/experiments/910b/20261123-gqa-teacher-student/round-06/RESULT.md)

## 全量同口径补测 / Matched full-coverage follow-up

2026-11-24 在同一 A2 上按用例用独立进程先运行 PyTorch golden，随后运行 R4。
两者使用同一计时入口、官方输入分布和预热采样设置。全量几何平均为
PyTorch **1.5010 ms** 与 R4 **230.4105 ms**，端到端延迟比 **153.51×**。
`gqa_8` 的配对数值为 **8.2383 ms** 与 **2821.5626 ms**。此次覆盖 20 例，
结果适用于该固定执行顺序和已记录的环境。

On 2026-11-24 both implementations were timed on the same A2 in separate
per-case processes, golden first and R4 second, using the same timing
entry point, public input distribution, warmups and sampling. The 20-case
geometric means are **1.5010 ms** for PyTorch and **230.4105 ms** for R4,
an end-to-end latency ratio of **153.51×**. The matched `gqa_8` values
are **8.2383 ms** and **2821.5626 ms**. The result applies to the recorded
environment and fixed execution order.

[全量配对样本 / Full paired samples](https://github.com/Charles-6701/forgekernel-ascend-repro/blob/d36e061aaa5af45d54f437c105834638b2058307/experiments/910b/20261123-gqa-teacher-student/pytorch-reference/full-paired-result.json)
