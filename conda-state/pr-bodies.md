# PR bodies (drafts for the user to post)

## 1. cppjit PR — branch `conda-support`

Title: [build] Add the license file and document the conda channel

Body:

> Adds the BSD-3-Clause-LBNL license file inherited from cppyy (PEP 639
> picks it up for future sdists and wheels; the conda-forge recipe
> requires it), a conda source-build section in the README, and the
> conda-forge release step in ReleaseProcess.md.
>
> Validation:
> - conda source build + JIT smoke validated in a fresh conda-forge
>   container env (LLVM 21.1.8, lld)
> - recipe built with rattler-build against the 0.1.0a1 sdist

## 2. conda-forge/staged-recipes PR — `recipes/cppjit/`

Title: Add cppjit

Body:

> cppjit generates Python bindings for C++ at runtime by JIT-compiling
> C++ through CppInterOp and LLVM. It is the successor of cppyy
> (compiler-research), rebuilt on stock LLVM releases.
>
> Two notes for review:
>
> cppjit has never had a stable release: 0.1.0a1 is its first published
> version (PyPI carries only this alpha). Following the precedent of
> packages that published prereleases on main before any stable release
> existed (e.g. black), this recipe targets the main label; once a
> stable release exists, later prereleases will move to the `dev` label
> via a `dev` branch on the feedstock.
>
> The recipe vendors CppInterOp as a second pinned source: 0.1.0a1
> requires a CppInterOp snapshot newer than the conda-forge cppinterop
> release (1.9.0). The feedstock will switch to the `cppinterop`
> package at the first compatible CppInterOp release — tracked in the
> recipe header comment. The upstream license file was added in
> compiler-research/cppjit#<N> (link the conda-support PR).
