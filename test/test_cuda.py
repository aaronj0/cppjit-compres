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

    def test03_launch_config_forms(self):
        """dim3 tuples and dynamic grid sizes launch without recompiling"""

        import cppjit

        n = 300  # deliberately not a multiple of the block size
        block = 128
        grid = (n + block - 1) // block
        dev = cppjit.gbl.cppjit_cuda_iota(n)
        cppjit.gbl.cppjit_cuda_scale[(grid, 1, 1), (block, 1)](dev, n, 2)
        cppjit.gbl.cppjit_cuda_scale[grid, block, 0](dev, n, 3)
        cppjit.gbl.cppjit_cuda_scale[grid, block, 0, 0](dev, n, 5)
        total = cppjit.gbl.cppjit_cuda_sum_free(dev, n)
        assert total == 30 * n * (n - 1) // 2

    def test04_shared_memory_and_stream(self):
        """launch with dynamic shared memory on an explicit stream"""

        import cppjit

        cppjit.cppdef("""
        __global__ void cppjit_cuda_block_sum(const int* v, long long* out,
                                              int n) {
            extern __shared__ int buf[];
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            buf[threadIdx.x] = i < n ? v[i] : 0;
            __syncthreads();
            if (threadIdx.x == 0) {
                long long s = 0;
                for (int j = 0; j < blockDim.x; j++) s += buf[j];
                atomicAdd((unsigned long long*)out, (unsigned long long)s);
            }
        }

        long long* cppjit_cuda_alloc_ll() {
            long long* p = nullptr;
            cudaMalloc(&p, sizeof(long long));
            cudaMemset(p, 0, sizeof(long long));
            return p;
        }

        long long cppjit_cuda_read_free_ll(long long* p) {
            long long v = 0;
            cudaDeviceSynchronize();
            cudaMemcpy(&v, p, sizeof(long long), cudaMemcpyDeviceToHost);
            cudaFree(p);
            return v;
        }

        unsigned long long cppjit_cuda_stream_create() {
            cudaStream_t s = nullptr;
            cudaStreamCreate(&s);
            return (unsigned long long)s;
        }

        void cppjit_cuda_stream_sync_destroy(unsigned long long s) {
            cudaStreamSynchronize((cudaStream_t)s);
            cudaStreamDestroy((cudaStream_t)s);
        }
        """)

        n, block = 256, 64
        grid = n // block
        dev = cppjit.gbl.cppjit_cuda_iota(n)
        out = cppjit.gbl.cppjit_cuda_alloc_ll()
        stream = cppjit.gbl.cppjit_cuda_stream_create()
        shared_bytes = 4 * block
        cppjit.gbl.cppjit_cuda_block_sum[grid, block, shared_bytes, stream](dev, out, n)
        cppjit.gbl.cppjit_cuda_stream_sync_destroy(stream)
        assert cppjit.gbl.cppjit_cuda_read_free_ll(out) == n * (n - 1) // 2
        cppjit.gbl.cppjit_cuda_sum_free(dev, n)

    @mark.xfail(
        reason="libdevice is not linked into incremental device code; the "
        "clang fix is pending upstream and no released LLVM carries it; "
        "math kernels silently no-op with the launch error visible only "
        "via cudaGetLastError"
    )
    def test05_device_math(self):
        """Kernels can call CUDA device math (sinf & co via libdevice)"""

        import cppjit

        cppjit.cppdef("""
        __global__ void cppjit_cuda_math(float* out) {
            out[threadIdx.x] = sinf(0.0f) + expf(0.0f);
        }

        float* cppjit_cuda_alloc_f(int n) {
            float* p = nullptr;
            cudaMalloc(&p, n * sizeof(float));
            cudaMemset(p, 0, n * sizeof(float));
            return p;
        }

        float cppjit_cuda_read_free_f(float* p) {
            float v = 0.0f;
            cudaDeviceSynchronize();
            cudaMemcpy(&v, p, sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(p);
            return v;
        }

        int cppjit_cuda_sticky_err() {
            cudaDeviceSynchronize();
            return (int)cudaGetLastError();
        }
        """)

        dev = cppjit.gbl.cppjit_cuda_alloc_f(64)
        cppjit.gbl.cppjit_cuda_math[1, 64](dev)
        assert cppjit.gbl.cppjit_cuda_sticky_err() == 0
        assert cppjit.gbl.cppjit_cuda_read_free_f(dev) == 1.0

    def test06_include_path_reaches_device(self, tmp_path):
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

    def test07_namespaced_kernel(self):
        """Kernels defined inside a namespace launch through their scope"""

        import cppjit

        cppjit.cppdef("""
        namespace cppjit_cuda_ns {
        __global__ void ns_scale(int* v, int n, int f) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) v[i] *= f;
        }
        }
        """)

        n = 128
        dev = cppjit.gbl.cppjit_cuda_iota(n)
        cppjit.gbl.cppjit_cuda_ns.ns_scale[2, 64](dev, n, 7)
        total = cppjit.gbl.cppjit_cuda_sum_free(dev, n)
        assert total == 7 * n * (n - 1) // 2

    def test08_nvrtc_module(self):
        """Load an nvrtc-compiled module and launch its kernel"""

        import cppjit
        import cppjit.cuda
        from pytest import raises

        cppjit.load_library("libnvrtc")
        cppjit.cppdef("""
        #include <nvrtc.h>

        std::string cppjit_cuda_nvrtc_ptx() {
            const char* src = "extern \\"C\\" __global__"
                              " void nv_scale(int* v, int n, int f) {"
                              "  int i = blockIdx.x * blockDim.x + threadIdx.x;"
                              "  if (i < n) v[i] *= f;"
                              "}";
            nvrtcProgram prog;
            if (nvrtcCreateProgram(&prog, src, "nv.cu", 0, nullptr, nullptr))
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
        ptx = cppjit.gbl.cppjit_cuda_nvrtc_ptx()
        assert ptx

        mod = cppjit.cuda.load_module(ptx)
        kern = mod.get_kernel("nv_scale", "int*, int, int")
        n = 192
        dev = cppjit.gbl.cppjit_cuda_iota(n)
        kern[3, 64](dev, n, 5)
        total = cppjit.gbl.cppjit_cuda_sum_free(dev, n)
        assert total == 5 * n * (n - 1) // 2

        with raises(TypeError):
            kern(dev, n, 5)  # a launch config is required
        with raises(RuntimeError):
            mod.get_kernel("no_such_kernel")
