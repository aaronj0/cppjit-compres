from pytest import mark
from support import HAS_CUDA_CORE, HAS_CUPY, HAS_TORCH, IS_CUDA

# Interop matrix: producers (CuPy array, torch tensor, cuda.core buffer,
# cppjit device buffer) x consumers (JIT'd kernel launch, driver-route
# kernel, cuda.compute algorithm). Every test double-gates on CUDA mode
# and on the producing package; optional deps live in requirements-cuda.txt.

pytestmark = mark.skipif(not IS_CUDA, reason="CUDA mode not enabled")


def cppjit_device_count():
    import cppjit

    if not hasattr(cppjit.gbl, "cppjit_cuda_interop_device_count"):
        cppjit.cppdef("""
        #include <cuda_runtime.h>

        int cppjit_cuda_interop_device_count() {
            int n = 0;
            cudaGetDeviceCount(&n);
            return n;
        }
        """)
    return cppjit.gbl.cppjit_cuda_interop_device_count()


def ensure_interop_kernels():
    """Shared device-side fixture: a scale kernel plus iota/readback."""
    import cppjit

    if hasattr(cppjit.gbl, "cppjit_cuda_interop_iota"):
        return
    cppjit.cppdef("""
    #include <cuda_runtime.h>

    __global__ void cppjit_cuda_interop_scale(int* v, int n, int f) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) v[i] *= f;
    }

    int* cppjit_cuda_interop_iota(int n) {
        int* host = new int[n];
        for (int i = 0; i < n; i++) host[i] = i;
        int* dev = nullptr;
        cudaMalloc(&dev, n * sizeof(int));
        cudaMemcpy(dev, host, n * sizeof(int), cudaMemcpyHostToDevice);
        delete[] host;
        return dev;
    }

    long long cppjit_cuda_interop_sum_free(int* dev, int n) {
        int* host = new int[n];
        cudaDeviceSynchronize();
        cudaMemcpy(host, dev, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(dev);
        long long s = 0;
        for (int i = 0; i < n; i++) s += host[i];
        delete[] host;
        return s;
    }

    // foreign buffers (CuPy, torch, ...) are read back WITHOUT freeing:
    // their owner frees them
    long long cppjit_cuda_interop_sum(unsigned long long dev, int n) {
        int* host = new int[n];
        cudaDeviceSynchronize();
        cudaMemcpy(host, (const void*)dev, n * sizeof(int),
                   cudaMemcpyDeviceToHost);
        long long s = 0;
        for (int i = 0; i < n; i++) s += host[i];
        delete[] host;
        return s;
    }
    """)


_interop_kernel = None


def nvrtc_scale_kernel():
    """An nvrtc-compiled `nv_iscale(int*, int, int)` through cppjit.cuda."""
    global _interop_kernel
    if _interop_kernel is not None:
        return _interop_kernel
    import cppjit
    import cppjit.cuda

    cppjit.load_library("libnvrtc")
    cppjit.cppdef("""
    #include <nvrtc.h>

    std::string cppjit_cuda_interop_nvrtc_ptx() {
        const char* src = "extern \\"C\\" __global__"
                          " void nv_iscale(int* v, int n, int f) {"
                          "  int i = blockIdx.x * blockDim.x + threadIdx.x;"
                          "  if (i < n) v[i] *= f;"
                          "}";
        nvrtcProgram prog;
        if (nvrtcCreateProgram(&prog, src, "nvi.cu", 0, nullptr, nullptr))
            return "";
        if (nvrtcCompileProgram(prog, 0, nullptr))
            return "";
        size_t n = 0;
        nvrtcGetPTXSize(prog, &n);
        std::string ptx(n, ' ');
        nvrtcGetPTX(prog, ptx.data());
        nvrtcDestroyProgram(&prog);
        return ptx;
    }
    """)
    ptx = cppjit.gbl.cppjit_cuda_interop_nvrtc_ptx()
    assert ptx
    mod = cppjit.cuda.load_module(ptx)
    _interop_kernel = mod.get_kernel("nv_iscale", "int*, int, int")
    return _interop_kernel


@mark.skipif(not HAS_CUPY, reason="cupy not installed")
class TestCuPyInterop:
    def test01_same_device_visible(self):
        """CuPy and the interpreter agree on the CUDA device set"""

        import cupy

        assert cupy.cuda.runtime.getDeviceCount() == cppjit_device_count() > 0

    def test02_kernel_consumes_cupy(self):
        """CuPy arrays pass straight into kernel launches (DLPack route)"""

        import cupy
        from pytest import raises

        ensure_interop_kernels()
        kern = nvrtc_scale_kernel()
        n = 256
        arr = cupy.arange(n, dtype=cupy.int32)
        kern[4, 64](arr, n, 3)
        cupy.cuda.runtime.deviceSynchronize()
        # readback via cupy's own D2H (asnumpy needs no cupy-side JIT,
        # which would require CUDA_PATH to point at a cupy-major toolkit)
        assert int(cupy.asnumpy(arr).sum()) == 3 * n * (n - 1) // 2

        with raises(TypeError):  # dtype mismatch: float32 into int*
            kern[4, 64](cupy.arange(n, dtype=cupy.float32), n, 3)

    def test03_view_keepalive_and_cai_route(self):
        """views hold their exporter alive; the CAI route matches DLPack"""

        import cppjit
        import cppjit.cuda
        import cupy

        ensure_interop_kernels()
        kern = nvrtc_scale_kernel()
        n = 192

        class CAIOnly:  # strips DLPack so view() exercises the CAI branch
            def __init__(self, a):
                self._a = a
                self.__cuda_array_interface__ = a.__cuda_array_interface__

        arr = cupy.arange(n, dtype=cupy.int32)
        v = cppjit.cuda.view(CAIOnly(arr))
        assert v.ptr == arr.data.ptr and v.typestr == "<i4"

        expect = 5 * n * (n - 1) // 2
        del arr  # the view's owner chain keeps the memory alive
        kern[3, 64](v, n, 5)
        assert cppjit.gbl.cppjit_cuda_interop_sum(v.ptr, n) == expect

    def test04_jit_kernel_consumes_cupy(self):
        """CuPy arrays pass into JIT'd __global__ kernel launches"""

        import cppjit
        import cupy

        ensure_interop_kernels()
        n = 256
        arr = cupy.arange(n, dtype=cupy.int32)
        cppjit.gbl.cppjit_cuda_interop_scale[4, 64](arr, n, 3)
        cupy.cuda.runtime.deviceSynchronize()
        assert int(cupy.asnumpy(arr).sum()) == 3 * n * (n - 1) // 2

    def test05_producer_stream_is_ordered(self):
        """work queued on the producer's stream precedes the launch"""

        import cppjit
        import cupy

        ensure_interop_kernels()
        n = 1 << 20  # large enough that the fill is still in flight
        with cupy.cuda.Stream():
            arr = cupy.ones(n, dtype=cupy.int32)
        cppjit.gbl.cppjit_cuda_interop_scale[(n + 255) // 256, 256](arr, n, 3)
        cupy.cuda.runtime.deviceSynchronize()
        assert int(cupy.asnumpy(arr).sum()) == 3 * n


@mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTorchInterop:
    def test01_same_device_visible(self):
        """torch and the interpreter agree on the CUDA device set"""

        import torch

        assert torch.cuda.is_available()
        assert torch.cuda.device_count() == cppjit_device_count() > 0

    def test02_stream_protocol_launch(self):
        """JIT'd kernels launch on a torch stream via __cuda_stream__"""

        import cppjit
        import torch

        ensure_interop_kernels()
        stream = torch.cuda.Stream()
        n = 128
        buf = cppjit.gbl.cppjit_cuda_interop_iota(n)
        cppjit.gbl.cppjit_cuda_interop_scale[2, 64, 0, stream](buf, n, 3)
        stream.synchronize()
        total = cppjit.gbl.cppjit_cuda_interop_sum_free(buf, n)
        assert total == 3 * n * (n - 1) // 2

    def test03_kernel_consumes_tensor(self):
        """torch tensors pass into kernel launches (DLPack route)"""

        import torch
        from pytest import raises

        kern = nvrtc_scale_kernel()
        n = 192
        t = torch.arange(n, dtype=torch.int32, device="cuda")
        kern[3, 64](t, n, 5)
        torch.cuda.synchronize()
        assert int(t.sum()) == 5 * n * (n - 1) // 2

        with raises(TypeError):  # dtype mismatch: float32 into int*
            kern[3, 64](torch.zeros(n, dtype=torch.float32, device="cuda"), n, 5)
        with raises(TypeError):  # host memory
            kern[3, 64](torch.arange(n, dtype=torch.int32), n, 5)

    def test04_view_of_tensor(self):
        """view() imports a torch tensor without copying it"""

        import cppjit.cuda
        import torch

        t = torch.arange(64, dtype=torch.float32, device="cuda")
        v = cppjit.cuda.view(t)
        assert v.ptr == t.data_ptr()
        assert v.shape == (64,) and v.typestr == "<f4"
        cppjit.cuda._check_view(v, "const float*")

    def test05_jit_kernel_consumes_tensor(self):
        """torch tensors pass into JIT'd __global__ kernel launches"""

        import cppjit
        import torch

        ensure_interop_kernels()
        n = 192
        t = torch.arange(n, dtype=torch.int32, device="cuda")
        cppjit.gbl.cppjit_cuda_interop_scale[3, 64](t, n, 5)
        torch.cuda.synchronize()
        assert int(t.sum()) == 5 * n * (n - 1) // 2


@mark.skipif(not HAS_CUDA_CORE, reason="cuda.core not installed")
class TestCudaCoreInterop:
    def test01_same_device_visible(self):
        """cuda.core and the interpreter agree on the CUDA device set"""

        try:
            from cuda.core import system
        except ImportError:  # pre-1.0 layout
            from cuda.core.experimental import system

        n = getattr(system, "num_devices", None)  # 1.2+: property
        if n is None:
            n = system.get_num_devices()  # 1.1.x
        assert n == cppjit_device_count() > 0

    def test02_stream_protocol_launch(self):
        """JIT'd kernels launch on a cuda.core Stream via __cuda_stream__"""

        import cppjit
        from cuda.core import Device

        dev = Device()
        dev.set_current()
        stream = dev.create_stream()

        ensure_interop_kernels()
        n = 192
        buf = cppjit.gbl.cppjit_cuda_interop_iota(n)
        cppjit.gbl.cppjit_cuda_interop_scale[3, 64, 0, stream](buf, n, 7)
        stream.sync()
        total = cppjit.gbl.cppjit_cuda_interop_sum_free(buf, n)
        assert total == 7 * n * (n - 1) // 2

    def test03_graph_builder_capture(self):
        """JIT'd kernels record into a cuda.core GraphBuilder and replay"""

        import cppjit
        from cuda.core import Device

        dev = Device()
        dev.set_current()

        ensure_interop_kernels()
        n = 128
        buf = cppjit.gbl.cppjit_cuda_interop_iota(n)
        # module load (deferred driver JIT) happens outside the capture;
        # f=1 leaves the values alone
        cppjit.gbl.cppjit_cuda_interop_scale[2, 64](buf, n, 1)

        gb = dev.create_graph_builder().begin_building()
        for _ in range(2):
            cppjit.gbl.cppjit_cuda_interop_scale[2, 64, 0, gb](buf, n, 2)
        graph = gb.end_building().complete()

        stream = dev.create_stream()
        graph.upload(stream)
        for _ in range(3):
            graph.launch(stream)  # each replay multiplies by 4
        stream.sync()
        graph.close()

        total = cppjit.gbl.cppjit_cuda_interop_sum_free(buf, n)
        assert total == 4**3 * n * (n - 1) // 2
