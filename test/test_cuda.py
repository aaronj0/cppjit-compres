from pytest import mark
from support import IS_CUDA


@mark.skipif(not IS_CUDA, reason="CUDA mode not enabled")
class TestCUDA:
    def setup_class(cls):
        import cppjit

        cppjit.cppdef("""
        #include <cuda_runtime.h>

        __global__ void cppjit_cuda_scale(int* v, int n, int f) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) v[i] *= f;
        }

        int* cppjit_cuda_iota(int n) {
            int* host = new int[n];
            for (int i = 0; i < n; i++) host[i] = i;
            int* dev = nullptr;
            cudaMalloc(&dev, n * sizeof(int));
            cudaMemcpy(dev, host, n * sizeof(int), cudaMemcpyHostToDevice);
            delete[] host;
            return dev;
        }

        long long cppjit_cuda_sum_free(int* dev, int n) {
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

    def test01_device_present(self):
        """The interpreter sees at least one CUDA device"""

        import cppjit

        cppjit.cppdef("""
        int cppjit_cuda_device_count() {
            int n = 0;
            cudaGetDeviceCount(&n);
            return n;
        }
        """)
        assert cppjit.gbl.cppjit_cuda_device_count() > 0

    def test02_kernel_launch(self):
        """Define a __global__ kernel and launch it from python"""

        import cppjit

        n = 256
        dev = cppjit.gbl.cppjit_cuda_iota(n)
        cppjit.gbl.cppjit_cuda_scale[4, 64](dev, n, 3)
        total = cppjit.gbl.cppjit_cuda_sum_free(dev, n)
        assert total == 3 * n * (n - 1) // 2

    def test03_include_path_reaches_device(self, tmp_path):
        """add_include_path is visible to the device-side parse"""

        import cppjit

        hdr = tmp_path / "cuda_inc_kernel.h"
        hdr.write_text("""
        __global__ void cppjit_cuda_add_one(int* v, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) v[i] += 1;
        }
        """)
        cppjit.add_include_path(str(tmp_path))
        cppjit.cppdef('#include "cuda_inc_kernel.h"')
        n = 128
        dev = cppjit.gbl.cppjit_cuda_iota(n)
        cppjit.gbl.cppjit_cuda_add_one[2, 64](dev, n)
        total = cppjit.gbl.cppjit_cuda_sum_free(dev, n)
        assert total == n * (n - 1) // 2 + n
