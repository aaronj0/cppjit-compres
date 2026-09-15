# Draft upstream issue (CppInterOp, cross-linking llvm-project)

Title: In-process JIT: .eh_frame DW.ref.__gxx_personality_v0 exceeds
Delta32 range in conda environments

Body (claims all measured; logs referenced from ~/cppjit-conda/logs/):

> Running cppjit (CppInterOp @ 9802d61, in-process clang::Interpreter)
> built against conda-forge LLVM (shared libLLVM, 21.1.8 and 22.1.x
> both reproduce), exception-bearing JIT modules fail once the session
> has accumulated enough incremental modules:
>
>     JIT session error: In graph incr_module_NNNN-jitted-objectbuffer,
>     section .eh_frame: relocation target 0x...a0
>     (DW.ref.__gxx_personality_v0) is out of range of Delta32 fixup at
>     address 0x... (<anonymous block> @ 0x... + 0x13)
>
> The wrapper compilation then fails and surfaces as a Python-level
> TypeError/ValueError per call. Measured deltas sit just past 2 GB
> (~0x8D3B_BE80). The first LinkGraph that materializes
> DW.ref.__gxx_personality_v0 defines it in the JITDylib; every later
> graph links it as an external symbol, so once the allocator's pages
> drift more than 2 GB from that first block, the CIE personality
> Delta32 cannot reach it.
>
> Reproduction: conda env (linux-64) with the cppjit conda package
> (vendored CppInterOp built against conda-forge llvmdev 21.1.8, lld);
> run the cppjit pytest core subset in one process (~4000 incremental
> modules). Failure counts are layout-dependent run to run (0-14
> failures for one test file alone; ~90-117 for the 7-file subset), on
> both clang 21 and 22 variants. The same suite is stable in the PyPI
> wheel layout (static-LLVM libclangCppInterOp) on the same machine
> across hundreds of runs, so the trigger is the address-space layout
> that conda's shared-library set produces, not the suite.
>
> No library involved exports DW.ref.__gxx_personality_v0 (checked
> with nm -D on libcppjit, libclangCppInterOp, libLLVM-21, libstdc++,
> python); the symbol is JIT-materialized.
>
> Possible fixes, in order of preference:
> 1. JITLink: emit/keep a per-graph local DW.ref block instead of
>    reusing the first graph's definition across the session (or use a
>    64-bit-safe encoding for the CIE personality slot).
> 2. CppInterOp: configure the in-process LLJIT with a reserving
>    JITLinkMemoryManager (slab allocation) so all graphs stay within
>    a 2 GB span — `clang::Interpreter::create(CI, LLJITBuilder)` is
>    already exported for this.

> Disabling ASLR (`setarch -R`) does not help — the failure persists,
> so the >2 GB gap is inherent to the environment's mapping layout,
> not randomization.
