# Hardware knowledge base (hw_kb)

An operator-agnostic library of GPU-certified hardware-interface recipes, one tree per kernel
language and, inside it, one tree per GPU architecture:

```
hw_kb/
  cutedsl/           # CuTe DSL runs (task language "cutedsl")
    README.md        # rules, layout and fill strategy
    TEMPLATE.md
    sm90/  sm100/
  cuda/              # CUDA C++ runs (task language "cuda")
    README.md
    TEMPLATE.md
    sm100/
```

A run reads and writes exactly one tree, `hw_kb/<language>/<arch>/`, picked by the task's `language`
and `arch` (`src/forgekernel/loop/lang_mode.py`). The two language trees are independent: a reading
certified with one language's probe is not carried over to the other. See each tree's README for
the rules.


## Regression review / 退化复核

应用 `hw_kb` 条目后若出现正确性、编译或运行稳定性、性能退化，必须转交人工审核和矫正。
暂停晋升候选及自动复用涉事条目，保留最后一个通过验证的基线和失败证据。人工审核应核对条目
适用范围、SDK 与硬件版本、测量口径及对照结果，决定收窄、修正或撤回条目。修正后须在目标
硬件上重新验证，人工确认后才可恢复自动使用。

When applying an entry causes a regression in correctness, compile or runtime stability,
or performance, require human review and correction. Hold candidate promotion and
automated reuse of the affected entry, retain the last validated baseline and the
failure evidence, and revalidate on target hardware before a human approves reuse.
