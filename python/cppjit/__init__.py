"""CppJIT — live C++/CUDA in Python via clang-repl + CppInterOp.

Normal use is unchanged:  import cppjit; cppjit.cppdef("..."); cppjit.gbl.fn(...)

CUDA: set CPPJIT_ENABLE_CUDA=1 (the tutorial Docker image does). On import we
then pre-create a --cuda clang-repl interpreter via CppInterOp's C API *before*
the backend loads, so the backend adopts it, and we install a CUDA-safe
cppjit.cppdef shim (the default cppdef path mis-resolves under --cuda). All of
this is internal: notebooks only ever call cppjit.cppdef / cppjit.gbl.
"""
import os as _os

_CUDA_ENABLED = False
_iop_cuda = None


def _enable_cuda():
    """Best-effort: create a --cuda interpreter before the backend loads.

    Returns True on success. Locates this package's own libclangCppInterOp.so
    (no env var needed); CUDA root + arch come from CUDA / CPPJIT_OFFLOAD_ARCH
    with sane defaults. Any failure (no GPU, no CUDA) returns False and leaves
    the normal non-CUDA import path intact.
    """
    global _iop_cuda
    flag = _os.environ.get("CPPJIT_ENABLE_CUDA", "0").lower()
    if flag in ("", "0", "false", "no"):
        return False
    import ctypes
    lib = _os.path.join(_os.path.dirname(__file__),
                        "cppyy_backend", "lib", "libclangCppInterOp.so")
    if not _os.path.exists(lib):
        return False
    cuda = _os.environ.get("CUDA", "/usr/local/cuda-12.9")
    arch = _os.environ.get("CPPJIT_OFFLOAD_ARCH", "sm_89")
    try:
        iop = ctypes.CDLL(lib, ctypes.RTLD_GLOBAL)
        iop.cppinterop_CreateInterpreter.restype = ctypes.c_void_p
        iop.cppinterop_CreateInterpreter.argtypes = [
            ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t]
        iop.cppinterop_Declare.restype = ctypes.c_int
        iop.cppinterop_Declare.argtypes = [
            ctypes.c_char_p, ctypes.c_bool, ctypes.c_void_p]

        def _argv(a):
            arr = (ctypes.c_char_p * len(a))(*[s.encode() for s in a])
            return arr, len(a)

        h_, hn = _argv(["-std=c++17", "-march=native"])
        g_, gn = _argv(["--cuda", f"--cuda-path={cuda}", f"--offload-arch={arch}"])
        if not iop.cppinterop_CreateInterpreter(h_, hn, g_, gn):
            return False
        _iop_cuda = iop
        return True
    except Exception:
        return False


_CUDA_ENABLED = _enable_cuda()
CUDA_ENABLED = _CUDA_ENABLED   # public: True when import created a --cuda interpreter


import contextlib as _contextlib


@_contextlib.contextmanager
def _silence_fd2():
    """Silence the C-level stderr (fd 2) for the duration of the block."""
    saved = _os.dup(2)
    null = _os.open(_os.devnull, _os.O_WRONLY)
    try:
        _os.dup2(null, 2)
        _os.close(null)
        yield
    finally:
        _os.dup2(saved, 2)
        _os.close(saved)


# Load the backend. Under --cuda its bootstrap emits a couple of benign clang
# parse-errors (CppInternal::Dispatch / cling::runtime type probes); silence
# fd 2 just for this import so notebooks start clean. Diagnostics from later
# cppjit.cppdef calls are unaffected (they run outside this window).
_import_ctx = _silence_fd2() if _CUDA_ENABLED else _contextlib.nullcontext()
with _import_ctx:
    from . import cppyy as _cppyy  # noqa: E402  (adopts the interpreter above)

for _name in getattr(_cppyy, "__all__", []):
    globals()[_name] = getattr(_cppyy, _name)
gbl = _cppyy.gbl
_backend = _cppyy._backend


if _CUDA_ENABLED:
    def cppdef(code, verbose=True):
        """Declare C++/CUDA to the live clang-repl interpreter.

        CUDA-safe replacement for the default cppdef (which mis-resolves the
        dispatch wrapper under --cuda); goes straight through the C API.
        """
        if _iop_cuda.cppinterop_Declare(code.encode(), False, None) != 0:
            raise RuntimeError("cppjit.cppdef: declaration failed")
        return True
