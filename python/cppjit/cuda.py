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

__all__ = ["load_module", "view", "CudaModule", "CudaKernel"]

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

    // Consumer-side stream ordering for __cuda_array_interface__ v3: the
    // consumer stream waits on an event recorded on the producer stream.
    // 1 and 2 are CUDA's reserved legacy/per-thread default-stream
    // handles and are valid as-is.
    inline int wait_on(unsigned long long producer,
                       unsigned long long consumer) {
        cudaEvent_t e = nullptr;
        int rc = (int)cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        if (rc)
            return rc;
        rc = (int)cudaEventRecord(e, (cudaStream_t)producer);
        if (!rc)
            rc = (int)cudaStreamWaitEvent((cudaStream_t)consumer, e, 0);
        cudaEventDestroy(e);
        return rc;
    }
    } // namespace __cppjit_nvrtc
    """)
    _ns = getattr(cppjit.gbl, "__cppjit_nvrtc")


def _raise_driver_error(doing):
    rc = _ns.last_error
    name = _ns.error_name(rc)
    raise RuntimeError(f"{doing} failed: {name} ({rc})")


_dlpack_ns = None  # the JIT'd __cppjit_dlpack namespace (fetched by string)


def _ensure_dlpack():
    global _dlpack_ns
    if _dlpack_ns is not None:
        return
    _ensure_helpers()
    cppjit.cppdef("""
    #include <Python.h>
    #include <string>
    #include <vector>

    namespace __cppjit_dlpack {
    // The DLPack ABI (dmlc/dlpack), declared inline: legacy and versioned.
    struct DLDevice { int32_t device_type; int32_t device_id; };
    struct DLDataType { uint8_t code; uint8_t bits; uint16_t lanes; };
    struct DLTensor {
        void* data;
        DLDevice device;
        int32_t ndim;
        DLDataType dtype;
        int64_t* shape;
        int64_t* strides;
        uint64_t byte_offset;
    };
    struct DLManagedTensor {
        DLTensor dl_tensor;
        void* manager_ctx;
        void (*deleter)(DLManagedTensor*);
    };
    struct DLPackVersion { uint32_t major; uint32_t minor; };
    struct DLManagedTensorVersioned {
        DLPackVersion version;
        void* manager_ctx;
        void (*deleter)(DLManagedTensorVersioned*);
        uint64_t flags;
        DLTensor dl_tensor;
    };

    struct ViewInfo {
        std::string error;  // non-empty = failure
        unsigned long long data = 0;
        std::vector<long long> shape;
        std::vector<long long> strides;  // element counts; empty = compact
        int device_type = 0;
        int device_id = 0;
        int code = 0, bits = 0, lanes = 0;
        bool readonly = false;
        bool versioned = false;
        unsigned long long managed = 0;  // handle for release()
    };

    // Claim the capsule (rename to used_*) and copy the tensor metadata;
    // release() must be called exactly once afterwards.
    inline ViewInfo consume(PyObject* capsule) {
        ViewInfo v;
        const DLTensor* t = nullptr;
        if (PyCapsule_IsValid(capsule, "dltensor_versioned")) {
            auto* m = (DLManagedTensorVersioned*)PyCapsule_GetPointer(
                capsule, "dltensor_versioned");
            if (m->version.major > 1) {
                v.error = "unsupported DLPack major version";
                return v;
            }
            v.versioned = true;
            v.readonly = m->flags & 1;  // DLPACK_FLAG_BITMASK_READ_ONLY
            v.managed = (unsigned long long)m;
            t = &m->dl_tensor;
            PyCapsule_SetName(capsule, "used_dltensor_versioned");
        } else if (PyCapsule_IsValid(capsule, "dltensor")) {
            auto* m =
                (DLManagedTensor*)PyCapsule_GetPointer(capsule, "dltensor");
            v.managed = (unsigned long long)m;
            t = &m->dl_tensor;
            PyCapsule_SetName(capsule, "used_dltensor");
        } else {
            v.error = "not a DLPack capsule";
            return v;
        }
        v.data = (unsigned long long)((char*)t->data + t->byte_offset);
        v.device_type = t->device.device_type;
        v.device_id = t->device.device_id;
        v.code = t->dtype.code;
        v.bits = t->dtype.bits;
        v.lanes = t->dtype.lanes;
        v.shape.assign(t->shape, t->shape + t->ndim);
        if (t->strides)
            v.strides.assign(t->strides, t->strides + t->ndim);
        return v;
    }

    inline void release(unsigned long long managed, bool versioned) {
        if (!managed)
            return;
        if (versioned) {
            auto* m = (DLManagedTensorVersioned*)managed;
            if (m->deleter)
                m->deleter(m);
        } else {
            auto* m = (DLManagedTensor*)managed;
            if (m->deleter)
                m->deleter(m);
        }
    }
    } // namespace __cppjit_dlpack
    """)
    _dlpack_ns = getattr(cppjit.gbl, "__cppjit_dlpack")


# DLPack dtype codes -> numpy typestr kinds (bfloat/opaque unsupported)
_DLPACK_KIND = {0: "i", 1: "u", 2: "f", 5: "c", 6: "b"}
# C parameter base types -> expected typestrs, for launch-time checks
_CTYPE_TYPESTR = {
    "float": "<f4",
    "double": "<f8",
    "int": "<i4",
    "unsigned int": "<u4",
    "long": "<i8",
    "unsigned long": "<u8",
    "long long": "<i8",
    "unsigned long long": "<u8",
    "short": "<i2",
    "unsigned short": "<u2",
    "signed char": "<i1",
    "unsigned char": "<u1",
    "bool": "|b1",
}


def _typestr(code, bits, lanes):
    kind = _DLPACK_KIND.get(code)
    if kind is None or lanes != 1:
        raise TypeError(f"unsupported DLPack dtype (code {code}, {lanes} lanes)")
    if kind == "b":
        return "|b1"
    return f"<{kind}{bits // 8}"


class _DeviceView:
    """A device-buffer view: pointer + layout, holding its exporter alive."""

    def __init__(
        self,
        ptr,
        shape,
        typestr,
        strides=None,
        readonly=False,
        owner=None,
        managed=0,
        versioned=False,
        capsule=None,
        stream=0,
    ):
        self.ptr = int(ptr)
        self.shape = tuple(shape)
        self.typestr = typestr
        self.strides = None if strides is None else tuple(strides)  # bytes
        self.readonly = bool(readonly)
        self._owner = owner
        self._capsule = capsule
        self._managed = managed
        self._versioned = versioned
        # A view is itself an exporter, so it can be handed to kernel
        # launches and to other libraries; built once, since producers
        # that rebuild this dict per access dominate the launch cost.
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": self.shape,
            "typestr": typestr,
            "data": (self.ptr, self.readonly),
            "strides": self.strides,
            "stream": stream if stream else 1,
        }

    def close(self):
        """Release the DLPack tensor (once); CAI views hold no resources."""
        if self._managed:
            _dlpack_ns.release(self._managed, self._versioned)
            self._managed = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def view(obj, stream=0):
    """A device view of a __dlpack__ or __cuda_array_interface__ exporter,
    ordered against `stream` (an int handle or __cuda_stream__ object)."""
    stream = _stream_handle(stream)
    if hasattr(obj, "__dlpack_device__"):
        return _view_dlpack(obj, stream)
    cai = getattr(obj, "__cuda_array_interface__", None)
    if cai is not None:
        return _view_cai(obj, cai, stream)
    raise TypeError(
        "expected a __dlpack__ or __cuda_array_interface__ exporter, "
        f"not '{type(obj).__name__}'"
    )


def _view_dlpack(obj, stream):
    _ensure_dlpack()
    device_type, _ = obj.__dlpack_device__()
    if device_type not in (2, 13):  # kDLCUDA, kDLCUDAManaged
        raise TypeError(
            "__dlpack_device__ reports non-CUDA memory; CUDA kernels take "
            "device buffers"
        )
    # DLPack: the consumer names its stream; cppjit's 0 (legacy default)
    # is spelled 1 in the protocol
    consumer = 1 if stream == 0 else stream
    try:
        capsule = obj.__dlpack__(stream=consumer, max_version=(1, 1))
    except TypeError:  # pre-1.0 exporter without max_version
        capsule = obj.__dlpack__(stream=consumer)
    info = _dlpack_ns.consume(capsule)
    error = str(info.error)
    if error:
        raise TypeError(f"DLPack import failed: {error}")
    itemsize = info.bits // 8
    strides = tuple(s * itemsize for s in info.strides) or None
    return _DeviceView(
        info.data,
        tuple(info.shape),
        _typestr(info.code, info.bits, info.lanes),
        strides=strides,
        readonly=info.readonly,
        owner=obj,
        managed=info.managed,
        versioned=info.versioned,
        capsule=capsule,
        stream=stream,
    )


def _same_stream(a, b):
    """0 and 1 both name the legacy default stream (2 is per-thread)."""
    return a == b or (a in (0, 1) and b in (0, 1))


def _view_cai(obj, cai, stream):
    _ensure_helpers()
    if cai.get("mask") is not None:
        raise TypeError("masked CUDA arrays are not supported")
    ptr, readonly = cai["data"]
    producer = cai.get("stream")
    if producer == 0:
        raise TypeError(
            "__cuda_array_interface__ stream 0 is disallowed by the protocol"
        )
    if producer is not None and not _same_stream(producer, stream):
        # CAI: the producer names its stream; order ours after it
        rc = _ns.wait_on(producer, stream)
        if rc:
            raise RuntimeError(f"stream ordering failed: cudaError={rc}")
    strides = cai.get("strides")
    return _DeviceView(
        ptr,
        tuple(cai["shape"]),
        cai["typestr"],
        strides=None if strides is None else tuple(strides),
        readonly=bool(readonly),
        owner=obj,
        stream=stream,
    )


def _check_view(v, param):
    base = param.rstrip("* ").strip()
    const = base.startswith("const ")
    if const:
        base = base[len("const ") :].strip()
    if v.readonly and not const:
        raise TypeError(f"read-only buffer passed for mutable parameter '{param}'")
    want = _CTYPE_TYPESTR.get(base)
    if want and v.typestr not in (want, "|" + want[1:]):
        raise TypeError(
            f"dtype mismatch: the buffer is '{v.typestr}', the parameter '{param}'"
        )


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
    # pointer parameters travel as raw 64-bit addresses so device pointers
    # from any producer (view(), cudaMalloc, other libraries) are accepted;
    # the args table below is bitwise identical either way
    ctypes = ["unsigned long long" if t.rstrip().endswith("*") else t for t in types]
    args = "".join(f", {t} __a{i}" for i, t in enumerate(ctypes))
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
    _dispatchers[key] = (getattr(cppjit.gbl, name), types)
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


def _stream_handle(spec):
    """The stream slot: an int handle or a __cuda_stream__ object."""
    if isinstance(spec, int):
        return spec
    proto = getattr(spec, "__cuda_stream__", None)
    if proto is None:
        raise TypeError(
            "the CUDA launch stream must be an int handle or provide __cuda_stream__"
        )
    info = proto() if callable(proto) else proto
    try:
        version, handle = info
    except (TypeError, ValueError):
        raise TypeError("__cuda_stream__ must provide (version, handle)") from None
    if version != 0:
        raise TypeError(f"unsupported __cuda_stream__ protocol version {version}")
    return int(handle)


def _launch_config(key):
    items = key if isinstance(key, tuple) else (key,)
    if not 2 <= len(items) <= 4:
        raise TypeError(_CONFIG_ERROR)
    shmem = items[2] if len(items) > 2 else 0
    stream = _stream_handle(items[3]) if len(items) > 3 else 0
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
        conv = list(args)
        holders = []  # views (and their exporters) live through the launch
        for i, a in enumerate(args):
            if isinstance(a, (int, float)):
                continue
            v = a if isinstance(a, _DeviceView) else None
            if v is None and (
                hasattr(a, "__cuda_array_interface__")
                or hasattr(a, "__dlpack_device__")
            ):
                v = view(a, stream=self._config[7])
            if v is not None:
                if i < len(k._types):
                    _check_view(v, k._types[i])
                conv[i] = v.ptr
                holders.append(v)
                continue
            if isinstance(a, (bytes, bytearray, memoryview)) or hasattr(
                a, "__array_interface__"
            ):
                raise TypeError(
                    "host memory passed to a CUDA kernel; CUDA kernels take "
                    "device buffers"
                )
            try:  # bound C++ pointers (low-level views, instances)
                conv[i] = int(cppjit.addressof(a))
                holders.append(a)
            except (TypeError, ValueError):
                pass  # leave to the dispatcher's own conversion
        rc = k._dispatch(k._handle, *self._config, *conv)
        del holders
        if rc:
            name = _ns.error_name(rc)
            raise RuntimeError(f"launching '{k._name}' failed: {name} ({rc})")


class CudaKernel:
    def __init__(self, module, name, handle, dispatch, types):
        self._module = module  # keeps the module alive
        self._name = name
        self._handle = handle
        self._dispatch = dispatch
        self._types = types

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
        dispatch, types = _dispatcher(params)
        handle = _ns.get_function(self._handle, name)
        if not handle:
            _raise_driver_error(f"resolving kernel '{name}'")
        return CudaKernel(self, name, handle, dispatch, types)

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
