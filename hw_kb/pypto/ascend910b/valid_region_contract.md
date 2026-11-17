# 带保护区的有效区域回写与互斥所有者访问计数 / Guarded valid-region writeback and disjoint-owner visit counters

- 架构 / Architecture：Ascend 910B；使用认证 job 所安装环境中的 PyPTO NPU 执行。设备、软件包版本和本地 API 签名均由 probe 记录在 job 日志中。 / Ascend 910B, PyPTO NPU execution in the certifying job's installed environment; device, package versions and local API signatures are recorded by the probe in the job log.
- 状态 / Status：已验证 / Verified — `PROBE PASS max_diff=1.1920928955078125e-07`，job `1794845325`（第 2 轮认证 / certification round 2）。
- 认证证据 / Certification evidence：`notes/valid_region_contract.py`，job `1794845325`；`probe_complete=true failures=0`。最大误差是完整 probe 的聚合值；可复用范围仅限于下述独立合成契约。 / The maximum is the complete probe's aggregate; the reusable scope is the independent synthetic contract below.

## 目的 / Purpose

本测试使用独立合成数据、哨兵保护区、输入保持检查和设备写入的访问计数张量，验证 `pypto.view` / `pypto.assemble` 的有效区域回写。每个互不重叠的矩形所有者从不可变 seed tile 读取、加一并写入独立的 visit 输出；seed 与 visit 不得别名，以避免编译图中的原地读写依赖。使用非原子计数器之前，必须先通过独立标量遍历证明所有者不重叠；同步后的数值检查用于验证重复 tile 工作和复用分配下的语义顺序。

Test valid-region `pypto.view` / `pypto.assemble` writeback using independent
synthetic data, sentinel guards, preserved inputs and a device-written visit
tensor. Each disjoint rectangular owner reads an immutable seed tile, adds one, and
writes a separate visit-output tile. Input seeds and output visits must never
alias. This avoids an in-place read/write dependency in the compiled graph.
Independent scalar traversal must establish non-overlap before using this
non-atomic counter. Numerical checks after synchronization test semantic
ordering across repeated tile work and reused allocations.

This is a contract test, not an atomic counting primitive or a measurement of
physical buffer allocation, stage count, overlap, stalls or speed.

这是契约测试，不是原子计数原语，也不测量物理缓冲区分配、流水级数、重叠、停顿或速度。

## 适用场景 / When needed

当固定容量 tile 超出有效边界、输出残留初始化值、相邻区域被意外修改，或重复分块工作产生陈旧值时使用。修改寻址、有效范围、dtype、布局或编译器选项后，应检查所有有效元素及其周围保护区。

Use when a fixed-capacity tile extends past a valid boundary, outputs retain
initialization values, adjacent regions change, or repeated tiled work produces
stale values. Check every valid element and surrounding guards after changing
addressing, valid extents, dtype, layout or compiler options.

The example scope is contiguous rank-2 FP32 tensors, 8 by 32 logical tiles,
two partial axes, inner unroll option 4 and buffer option 8. These are synthetic
test parameters, not recommended tuning values. Other ranks and dtypes need
their own recorded scope. A successful larger application test does not turn
this small example into a universal memory-safety proof.

示例范围为连续 rank-2 FP32 张量、8×32 逻辑 tile、两个不完整轴、内部展开选项 4 和缓冲选项 8。这些只是合成测试参数，不是调优建议。其他 rank 和 dtype 必须记录自己的认证范围；更大的应用测试通过也不能把该小型示例提升为通用内存安全证明。

## 最小可运行示例 / Minimal runnable example

请使用目标环境配置的 PyPTO、torch 和 torch_npu 解释器运行。被测设备运算与回写全部使用 PyPTO；torch 仅用于分配输入和保护区，独立期望值在 CPU 上计算。

Run this with the target's configured PyPTO, torch and torch_npu interpreter.
All tested device arithmetic and writeback use PyPTO. Torch allocates inputs
and guard storage; the independent expected values are computed on the CPU.

```python
import math
import torch
import torch_npu
import pypto


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU, "valid_shape_optimize": 1},
    pass_options={"vec_nbuffer_setting": {"DEFAULT": 8}},
)
def synthetic_contract(
    a: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    b: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    seeds: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    counts: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8, 32)
    for i in pypto.loop((a.shape[0] + 7) // 8, name="row"):
        for j in pypto.loop((a.shape[1] + 31) // 32, name="col", unroll_list=[4]):
            r = i * 8
            c = j * 32
            vr = (a.shape[0] - r).min(8)
            vc = (a.shape[1] - c).min(32)
            av = pypto.view(a, [8, 32], [r, c], valid_shape=[vr, vc])
            bv = pypto.view(b, [8, 32], [r, c], valid_shape=[vr, vc])
            value = av + bv * 2.0
            store = pypto.view(value, [8, 32], [0, 0], valid_shape=[vr, vc])
            pypto.assemble(store, [r, c], y)
            old = pypto.view(seeds, [8, 32], [r, c], valid_shape=[vr, vc])
            new = old + 1.0
            stamp = pypto.view(new, [8, 32], [0, 0], valid_shape=[vr, vc])
            pypto.assemble(stamp, [r, c], counts)

torch.npu.set_device(0)
shape = (83, 131)
n, guard, sentinel = 83 * 131, 32768, -37.0

def allocate(fill):
    base = torch.full((n + 2 * guard,), sentinel, dtype=torch.float32, device="npu:0")
    logical = base[guard:guard + n].view(shape)
    logical.fill_(fill)
    return logical, base

seq = torch.arange(n, dtype=torch.float32).reshape(shape)
a_cpu = (seq.remainder(113) - 56) / 16
b_cpu = (seq.remainder(79) + 17) / 32
a, ga = allocate(0)
b, gb = allocate(0)
y, gy = allocate(sentinel)
seeds, gs = allocate(0)
visits, gv = allocate(sentinel)
a.copy_(a_cpu)
b.copy_(b_cpu)
synthetic_contract(a, b, y, seeds, visits)
torch_npu.npu.synchronize()
errors = {
    "value": (y.cpu() - (a_cpu + 2 * b_cpu)).abs().max().item(),
    "visits": (visits.cpu() - 1).abs().max().item(),
    "input-a": (a.cpu() - a_cpu).abs().max().item(),
    "input-b": (b.cpu() - b_cpu).abs().max().item(),
    "counter-input": seeds.cpu().abs().max().item(),
}
for name, base in zip(("a", "b", "y", "seeds", "visits"), (ga, gb, gy, gs, gv)):
    for side, edge in (("prefix", base[:guard]), ("suffix", base[-guard:])):
        errors[name + "/" + side] = (edge.cpu() - sentinel).abs().max().item()
for name, error in errors.items():
    passed = math.isfinite(error) and error == 0
    print(f"PROBE {'PASS' if passed else 'FAIL'} max_diff={error} contract={name}")

```

The standalone certification probe additionally changes both inputs while
reusing storage, uses zero seeds and sentinel-filled visit outputs for each normal run, and
executes a second ordered launch with the previous visits as input and a fresh
destination to check that counts become two. The
all-one acceptance check must reject that duplicate-visit control.

独立认证 probe 还会在复用存储的同时改变两个输入；每次正常运行使用全零 seeds 和哨兵填充的 visit 输出；随后把前一次 visits 作为输入、使用新目的张量执行有序的第二次 launch，并验证计数变为 2。全 1 接受条件必须拒绝该重复访问对照。

## 注意事项 / Gotchas

- The counter reads seeds and writes a separate destination. Concurrent
  overlapping owners can overwrite identical values and escape detection.
  Such mappings are a **no-go** for this recipe; an
  all-one result alone does not prove absence of racing duplicate writers.
- In-place counting is a **no-go** in this recipe. An earlier attempted
  in-place read/add/assemble graph failed before launch with
  `OperationLoopCheck: Loop Detected`; it produced no numerical verdict.
  Separate allocations were the retry change certified by job 1794845325.
  This does not establish a general PyPTO in-place limitation.
- A scalar coverage check is a precondition, not device evidence. Device
  counts, guards, unchanged inputs and numerical values are separate checks.
- Counter instrumentation changes the compiled graph. Validate repeated data
  through the original graph separately before attributing ordering semantics
  to an application. Do not attribute its scheduling to the instrumented copy.
- Invalid tile storage has no certified value. Clamp every valid extent and
  reapply it to the final store operand after arithmetic. Do not infer padded
  read values from a successful masked store.
- Guards observe only the allocated prefix/suffix regions; they do not catch
  arbitrary reads or all internal aliasing. Full valid-output comparison is
  also required. Allocations must not overlap inputs, outputs or counters.
- More logical tiles than a buffer option exercises repeated work. It does
  not establish that the option equals physical stages or prove how many
  iterations ran on each core.
- A build or launch exception produces no numerical verdict. A launch-only
  `max_diff=n/a` cannot discharge any value, preservation, guard or visit line.

中文要点：

- 计数器读取 seeds 并写入独立目的张量。存在并发重叠所有者时，非原子写可能以相同值相互覆盖而漏检；此类映射禁止使用本方案，结果全 1 不能单独证明无竞态重复写。
- 禁止原地计数。先前的原地 read/add/assemble 图在 launch 前触发 `OperationLoopCheck: Loop Detected`，没有数值裁决。job `1794845325` 认证的是输入、输出分离的重试方案，这不代表 PyPTO 普遍不支持原地操作。
- 标量覆盖检查只是前置条件，不是设备证据。设备计数、保护区、输入不变性和数值结果必须分别检查。
- 插桩会改变编译图；应使用原始未插桩图单独验证重复数据，不得把插桩副本的调度属性归因于应用实现。
- 无效 tile 存储区没有认证值。必须夹紧每个有效范围，并在算术运算后将其重新应用到最终 store 操作数。
- 保护区只能观察分配前缀/后缀，不能发现任意越界读取或所有内部别名；仍需比较完整有效输出，并保证输入、输出和计数器分配互不重叠。
- 逻辑 tile 数超过缓冲选项只能说明发生重复工作，不能证明物理流水级数或每核迭代次数。
- 构建或 launch 异常不产生数值裁决；仅有 `max_diff=n/a` 不能满足任何数值、保持、保护区或访问检查。

## 重试前提 / Retry preconditions

若出现非有限值或 value/guard/count 契约失败，应先隔离失败项并修复映射、有效范围、输入模式或依赖关系，禁止原样重复提交以期待不同结果。若编译或 launch 失败，应先修正已安装 API、依赖图或受支持布局，并在记录中保留“无数值结果”。

For nonfinite values or a failed value/guard/count contract, isolate the
failing line and fix the mapping, valid extent, input pattern or dependence
before resubmission. Do not repeat unchanged code expecting another verdict.
For a compile/launch failure, correct the installed API, dependency graph or
supported layout first and retain the absence of a numerical result in the record.

Overlapping ownership requires a separately certified atomic counter or
per-owner observation scheme. New hardware, SDK, dtype, rank, storage layout,
geometry, buffer or unroll settings require a new scoped driver job. Without
the target driver or its completed PASS, keep this entry pending.

所有者重叠需要单独认证的原子计数器或逐所有者观察机制。更换硬件、SDK、dtype、rank、存储布局、几何形状、缓冲或展开设置均须重新运行限定范围的 driver job；缺少目标 driver 或完整 PASS 时，本条目必须保持 pending。

## 来源 / Sources

- Public API symbols: `pypto.frontend.jit`, `pypto.RunMode.NPU`,
  `pypto.Tensor`, `pypto.DYNAMIC`, `pypto.DT_FP32`, `pypto.loop`,
  `pypto.set_vec_tile_shapes`, `pypto.view`, `pypto.assemble` and scalar `.min`.
- The probe records signatures and installed local API module paths for
  `view`, `assemble`, `loop` and `set_vec_tile_shapes` before execution.
- Certification probe: `notes/valid_region_contract.py`; PROBE PASS max_diff=1.1920928955078125e-07, job 1794845325 (round 2), all contract lines passed and the job completed.

中文摘要：可复用声明仅由公开 API 符号和认证 probe 支撑。probe 在执行前记录 `view`、`assemble`、`loop` 与 `set_vec_tile_shapes` 的签名及本地模块路径；认证 job 为 `1794845325`（第 2 轮），所有契约行通过且 job 正常完成。

## 认证范围 / Certification scope

job `1794845325` 仅认证合成 rank-2 FP32 机制：连续 83×131 输入、8×32 逻辑 tile、55 个互不重叠所有者、两个不完整轴、`valid_shape_optimize=1`、缓冲选项 8、内部展开选项 4。完整 probe 检查独立 CPU 期望值、输入和 counter seeds 不变性、前后缀保护区、设备全 1 visits、改变输入并复用分配的重复 launch，以及其全 2 visits 必须被全 1 检测器拒绝的有序重复对照。全部契约行通过，聚合最大有限误差为 `1.1920928955078125e-07`，失败数为 0。

Job 1794845325 certifies the synthetic rank-2 FP32 mechanism: contiguous
83 by 131 inputs, 8 by 32 logical tiles, 55 disjoint owners, two partial
axes, `valid_shape_optimize=1`, buffer option 8 and inner unroll option 4.
The completed probe checks values against independent CPU expectations,
unchanged inputs and counter seeds, prefix/suffix guards, all-one device
visits, repeated launches with changed inputs and reused allocations, and
an ordered duplicate control whose all-two visits must be rejected by the
all-one detector. Every contract line passed; the complete probe reported
finite maximum error 1.1920928955078125e-07 and zero failures.

Certification is limited to the installed environment recorded in that
job's device/package/API readings. It does not transfer to other software
versions, ranks, dtypes, geometries or layouts without a new scoped job.
Application contracts remain separate evidence. No physical UB/L1
allocation, hardware slot count, overlap, stall or performance claim follows.

认证仅绑定到该 job 记录的设备、软件包和 API 环境；若软件版本、rank、dtype、几何形状或布局变化，必须重新认证。应用契约仍需独立证据。本条目不对物理 UB/L1 分配、硬件 slot 数、执行重叠、停顿或性能作任何声明。
