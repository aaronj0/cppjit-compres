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
