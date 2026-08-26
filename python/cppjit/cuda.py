"""Driver-level CUDA interop: load modules compiled outside the
interpreter (nvrtc PTX, ptxas cubins) and launch their kernels with the
same spelling as JIT'd __global__ kernels:

    >>> import cppjit.cuda
    >>> mod = cppjit.cuda.load_module(ptx)          # str or bytes image
    >>> kern = mod.get_kernel("saxpy", "int, float, const float*, float*")
    >>> kern[grid, block](n, a, x, y)               # [grid, block(, shared_bytes(, stream))]

The CUDA runtime the interpreter uses and the driver API share the
device's primary context, so pointers from cudaMalloc and cudaStream_t
handles flow freely between JIT'd and loaded kernels.

PTX carries parameter sizes but not C types, so get_kernel takes the
kernel's parameter list; a per-signature dispatch function is JIT'd
once and reused by every kernel and module with that signature.
"""

import cppjit

__all__ = ["load_module", "CudaModule", "CudaKernel"]

_CONFIG_ERROR = (
    "CUDA kernels take a launch config: [grid, block(, shared_bytes(, stream))]"
)

_ns = None  # the JIT'd __cppjit_nvrtc namespace (a dunder name mangles
# inside class bodies, so it is fetched once by string here)
_dispatchers = {}  # normalized parameter list -> JIT'd dispatch function


def _ensure_helpers():
    global _ns
    if _ns is not None:
        return
    cppjit.load_library("libcuda")
    cppjit.cppdef("""
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <string>

    namespace __cppjit_nvrtc {
    inline int last_error = 0;

    inline std::string error_name(int rc) {
        const char* s = nullptr;
        cuGetErrorName((CUresult)rc, &s);
        return s ? s : "unknown CUDA driver error";
    }

    // The runtime owns the primary context; a first runtime call both
    // creates it and makes it current, so driver-API loads land in the
    // same context JIT'd kernels launch in.
    inline bool ensure_context() {
        if (cudaFree(nullptr) != cudaSuccess)
            return false;
        CUcontext ctx = nullptr;
        cuCtxGetCurrent(&ctx);
        if (ctx)
            return true;
        int dev = 0;
        cudaGetDevice(&dev);
        if ((last_error = (int)cuDevicePrimaryCtxRetain(&ctx, dev)))
            return false;
        return !(last_error = (int)cuCtxSetCurrent(ctx));
    }

    inline unsigned long long load_module(const std::string& image) {
        if (!ensure_context())
            return 0;
        CUmodule mod = nullptr;
        // std::string guarantees the NUL terminator PTX text needs.
        last_error = (int)cuModuleLoadData(&mod, image.c_str());
        return last_error ? 0 : (unsigned long long)mod;
    }

    inline unsigned long long get_function(unsigned long long mod,
                                           const std::string& name) {
        CUfunction fn = nullptr;
        last_error =
            (int)cuModuleGetFunction(&fn, (CUmodule)mod, name.c_str());
        return last_error ? 0 : (unsigned long long)fn;
    }

    inline void unload_module(unsigned long long mod) {
        cuModuleUnload((CUmodule)mod);
    }
    } // namespace __cppjit_nvrtc
    """)
    _ns = getattr(cppjit.gbl, "__cppjit_nvrtc")


def _raise_driver_error(doing):
    rc = _ns.last_error
    name = _ns.error_name(rc)
    raise RuntimeError(f"{doing} failed: {name} ({rc})")


def _split_params(params):
    """Split a C++ parameter list on top-level commas."""
    parts, depth, start = [], 0, 0
    for i, c in enumerate(params):
        if c in "<([":
            depth += 1
        elif c in ">)]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(params[start:i])
            start = i + 1
    parts.append(params[start:])
    parts = [p.strip() for p in parts]
    return [] if parts == [""] or parts == ["void"] else parts


def _dispatcher(params):
    """JIT (once per signature) a typed cuLaunchKernel dispatch function."""
    types = _split_params(params)
    for t in types:
        if "&" in t:
            raise ValueError(f"kernel parameters pass by value, not '{t}'")
    key = ",".join(types)
    if key in _dispatchers:
        return _dispatchers[key]

    name = f"__cppjit_nvrtc_launch_{len(_dispatchers)}"
    args = "".join(f", {t} __a{i}" for i, t in enumerate(types))
    if types:
        table = ", ".join(f"(void*)&__a{i}" for i in range(len(types)))
        argv = f"void* __args[] = {{{table}}};"
    else:
        argv = "void** __args = nullptr;"
    cppjit.cppdef(f"""
    int {name}(unsigned long long __fn, unsigned __gx, unsigned __gy,
               unsigned __gz, unsigned __bx, unsigned __by, unsigned __bz,
               unsigned long long __shmem, unsigned long long __stream{args}) {{
        {argv}
        return (int)cuLaunchKernel((CUfunction)__fn, __gx, __gy, __gz, __bx,
                                   __by, __bz, (unsigned)__shmem,
                                   (CUstream)__stream, __args, nullptr);
    }}""")
    _dispatchers[key] = getattr(cppjit.gbl, name)
    return _dispatchers[key]


def _dims(spec, what):
    """One grid/block spec: an int or a sequence of up to three ints."""
    seq = [spec] if isinstance(spec, int) else list(spec)
    if not 1 <= len(seq) <= 3:
        raise ValueError(f"{what} takes one to three dimensions")
    for v in seq:
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"{what} dimensions must be positive")
    return tuple(seq) + (1,) * (3 - len(seq))


def _launch_config(key):
    items = key if isinstance(key, tuple) else (key,)
    if not 2 <= len(items) <= 4:
        raise TypeError(_CONFIG_ERROR)
    shmem = items[2] if len(items) > 2 else 0
    stream = items[3] if len(items) > 3 else 0
    return _dims(items[0], "grid") + _dims(items[1], "block") + (shmem, stream)


class _BoundKernel:
    """A kernel with its launch config bound, ready to call."""

    def __init__(self, kernel, config):
        self._kernel = kernel
        self._config = config

    def __call__(self, *args, **kwds):
        if kwds:
            raise TypeError("CUDA kernel launches take no keyword arguments")
        k = self._kernel
        rc = k._dispatch(k._handle, *self._config, *args)
        if rc:
            name = _ns.error_name(rc)
            raise RuntimeError(f"launching '{k._name}' failed: {name} ({rc})")


class CudaKernel:
    def __init__(self, module, name, handle, dispatch):
        self._module = module  # keeps the module alive
        self._name = name
        self._handle = handle
        self._dispatch = dispatch

    def __getitem__(self, key):
        return _BoundKernel(self, _launch_config(key))

    def __call__(self, *args, **kwds):
        raise TypeError(_CONFIG_ERROR)


class CudaModule:
    def __init__(self, handle):
        self._handle = handle

    def get_kernel(self, name, params=""):
        """Resolve kernel `name` taking the C++ parameter list `params`."""
        if not self._handle:
            raise RuntimeError("the module has been unloaded")
        dispatch = _dispatcher(params)
        handle = _ns.get_function(self._handle, name)
        if not handle:
            _raise_driver_error(f"resolving kernel '{name}'")
        return CudaKernel(self, name, handle, dispatch)

    def unload(self):
        """Unload the module; its kernels become unlaunchable."""
        if self._handle:
            _ns.unload_module(self._handle)
            self._handle = 0


def load_module(image):
    """Load a PTX (str or bytes) or cubin (bytes) image into the CUDA
    context the interpreter uses and return a CudaModule."""
    _ensure_helpers()
    if isinstance(image, str):
        image = image.encode()
    handle = _ns.load_module(image)
    if not handle:
        _raise_driver_error("loading the module")
    return CudaModule(handle)
