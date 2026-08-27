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
    """)


@mark.skipif(not HAS_CUPY, reason="cupy not installed")
class TestCuPyInterop:
    def test01_same_device_visible(self):
        """CuPy and the interpreter agree on the CUDA device set"""

        import cupy

        assert cupy.cuda.runtime.getDeviceCount() == cppjit_device_count() > 0


@mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTorchInterop:
    def test01_same_device_visible(self):
        """torch and the interpreter agree on the CUDA device set"""

        import torch

        assert torch.cuda.is_available()
        assert torch.cuda.device_count() == cppjit_device_count() > 0


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
