# Release Notes

This document contains the release notes for cppjit, release 0.1.0a1.
cppjit embeds an interactive C++ JIT compiler in Python: write or import
C++ at runtime and use its functions, classes, and templates as if they
were Python. cppjit is the successor project of
[cppyy](https://github.com/wlav/cppyy), rebuilt on
[CppInterOp](https://github.com/compiler-research/CppInterOp) and the
clang-repl C++ interpreter in LLVM. In a git checkout this document
describes the release under development; the notes of released versions
are archived on the
[releases page](https://github.com/compiler-research/cppjit/releases).

## What's New in cppjit 0.1.0a1

The first prerelease of cppjit. This release establishes the build,
test, and packaging infrastructure.

- Runtime C++ bindings: `cppdef`, `include`, and `load_library` build
  Python proxies for functions, classes, and templates on demand, with
  lazy JIT compilation of the C++ they touch.
- Wheels for Linux x86_64/aarch64 and macOS arm64/x86_64 (Python
  3.12-3.14, LLVM 21 toolchain), with the full test suite run inside
  the wheel build.
- An external CppInterOp, either an install prefix or a build
  directory, is consumed in place through `find_package`
  (`CppInterOp_DIR`); the default build bundles the pinned one.
- A plain CMake build yields a runnable tree with no install step;
  `pip`, editable `pip`, and CMake development flows are documented in
  the README.
- CI covers every supported Python on both supported LLVM majors
  (21, 22), C++17/20/23, Linux x86_64/arm64 and both macOS
  architectures, with valgrind cells and a scheduled nightly superset.
- std::filesystem and std::span tests are capability-gated: on hosts
  whose libstdc++ runtime predates the features (manylinux base
  images), the suite skips them instead of crashing the JIT.

## External Dependencies

- CppInterOp pin: 9802d61
- LLVM/Clang 21-22; Python 3.12-3.14; Linux and macOS

## Known Issues

- The Linux arm64 valgrind nightly cell reports JIT-related false
  positives pending symbolized suppressions; tracked for the next
  release.
- Windows is not yet supported.
