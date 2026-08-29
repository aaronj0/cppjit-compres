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

    def test09_stream_protocol(self):
        """The stream slot accepts __cuda_stream__ objects"""

        import cppjit
        from pytest import raises

        cppjit.cppdef("""
        unsigned long long cppjit_cuda_sp_stream_create() {
            cudaStream_t s = nullptr;
            cudaStreamCreate(&s);
            return (unsigned long long)s;
        }
        void cppjit_cuda_sp_stream_sync_destroy(unsigned long long s) {
            cudaStreamSynchronize((cudaStream_t)s);
            cudaStreamDestroy((cudaStream_t)s);
        }
        """)
        handle = cppjit.gbl.cppjit_cuda_sp_stream_create()

        class MethodStream:
            def __cuda_stream__(self):
                return (0, handle)

        class TupleStream:
            pass

        tuple_stream = TupleStream()
        tuple_stream.__cuda_stream__ = (0, handle)

        n = 128
        dev = cppjit.gbl.cppjit_cuda_iota(n)
        cppjit.gbl.cppjit_cuda_scale[2, 64, 0, MethodStream()](dev, n, 3)
        cppjit.gbl.cppjit_cuda_scale[2, 64, 0, tuple_stream](dev, n, 5)
        cppjit.gbl.cppjit_cuda_sp_stream_sync_destroy(handle)

        total = cppjit.gbl.cppjit_cuda_sum_free(dev, n)
        assert total == 15 * n * (n - 1) // 2

        class WrongVersion:
            def __cuda_stream__(self):
                return (1, 0)

        with raises(TypeError):
            cppjit.gbl.cppjit_cuda_scale[2, 64, 0, object()]
        with raises(TypeError):
            cppjit.gbl.cppjit_cuda_scale[2, 64, 0, WrongVersion()]

    def test10_stream_protocol_python_surface(self):
        """cppjit.cuda mirrors the __cuda_stream__ acceptance"""

        import cppjit.cuda
        from pytest import raises

        class MethodStream:
            def __cuda_stream__(self):
                return (0, 1234)

        cfg = cppjit.cuda._launch_config((2, 64, 0, MethodStream()))
        assert cfg == (2, 1, 1, 64, 1, 1, 0, 1234)

        class WrongVersion:
            def __cuda_stream__(self):
                return (2, 0)

        with raises(TypeError):
            cppjit.cuda._launch_config((2, 64, 0, object()))
        with raises(TypeError):
            cppjit.cuda._launch_config((2, 64, 0, WrongVersion()))

    def test11_graph_capture(self):
        """python-driven launches record into a CUDA graph and replay"""

        import cppjit

        cppjit.cppdef("""
        unsigned long long cppjit_cuda_gc_stream() {
            cudaStream_t s = nullptr;
            cudaStreamCreate(&s);
            return (unsigned long long)s;
        }
        void cppjit_cuda_gc_begin(unsigned long long s) {
            cudaStreamBeginCapture((cudaStream_t)s,
                                   cudaStreamCaptureModeGlobal);
        }
        unsigned long long cppjit_cuda_gc_end(unsigned long long s) {
            cudaGraph_t g = nullptr;
            cudaStreamEndCapture((cudaStream_t)s, &g);
            return (unsigned long long)g;
        }
        unsigned long long cppjit_cuda_gc_instantiate(unsigned long long g) {
            cudaGraphExec_t e = nullptr;
            cudaGraphInstantiate(&e, (cudaGraph_t)g, 0);
            return (unsigned long long)e;
        }
        int cppjit_cuda_gc_replay(unsigned long long e, unsigned long long s,
                                  int reps) {
            for (int i = 0; i < reps; i++)
                cudaGraphLaunch((cudaGraphExec_t)e, (cudaStream_t)s);
            cudaStreamSynchronize((cudaStream_t)s);
            return (int)cudaGetLastError();
        }
        void cppjit_cuda_gc_destroy(unsigned long long e, unsigned long long g,
                                    unsigned long long s) {
            cudaGraphExecDestroy((cudaGraphExec_t)e);
            cudaGraphDestroy((cudaGraph_t)g);
            cudaStreamDestroy((cudaStream_t)s);
        }
        """)

        n = 128
        dev = cppjit.gbl.cppjit_cuda_iota(n)
        stream = cppjit.gbl.cppjit_cuda_gc_stream()
        kern = cppjit.gbl.cppjit_cuda_scale[2, 64, 0, stream]
        # a first launch outside the capture window loads the module (the
        # deferred driver JIT is not capture-safe); f=1 leaves values alone
        cppjit.gbl.cppjit_cuda_scale[2, 64, 0, stream](dev, n, 1)

        cppjit.gbl.cppjit_cuda_gc_begin(stream)
        for _ in range(3):
            kern(dev, n, 2)  # -> one replay multiplies by 8
        graph = cppjit.gbl.cppjit_cuda_gc_end(stream)
        gexec = cppjit.gbl.cppjit_cuda_gc_instantiate(graph)
        assert cppjit.gbl.cppjit_cuda_gc_replay(gexec, stream, 4) == 0
        cppjit.gbl.cppjit_cuda_gc_destroy(gexec, graph, stream)

        total = cppjit.gbl.cppjit_cuda_sum_free(dev, n)
        assert total == 2**12 * n * (n - 1) // 2

    def test12_device_views(self):
        """view() imports device buffers and rejects host memory"""

        import cppjit
        import cppjit.cuda
        from pytest import raises

        n = 64
        dev = cppjit.gbl.cppjit_cuda_iota(n)
        ptr = int(cppjit.addressof(dev))

        class CAIProducer:
            def __init__(self, readonly=False, stream=None):
                self.__cuda_array_interface__ = {
                    "version": 3,
                    "shape": (n,),
                    "typestr": "<i4",
                    "data": (ptr, readonly),
                    "strides": None,
                    "stream": stream,
                }

        v = cppjit.cuda.view(CAIProducer())
        assert v.ptr == ptr and v.shape == (n,) and v.typestr == "<i4"
        assert not v.readonly and v.strides is None

        # readonly propagates and gates mutable parameters
        ro = cppjit.cuda.view(CAIProducer(readonly=True))
        assert ro.readonly
        cppjit.cuda._check_view(ro, "const int*")
        with raises(TypeError):
            cppjit.cuda._check_view(ro, "int*")
        with raises(TypeError):
            cppjit.cuda._check_view(v, "float*")  # dtype mismatch

        # protocol violations and host memory are rejected
        with raises(TypeError):
            cppjit.cuda.view(CAIProducer(stream=0))  # 0 is disallowed
        with raises(TypeError):
            cppjit.cuda.view(object())
        import numpy as np

        with raises(TypeError):  # numpy speaks DLPack, but for host memory
            cppjit.cuda.view(np.arange(4))

        cppjit.gbl.cppjit_cuda_sum_free(dev, n)

    def test13_optimized_by_default(self):
        """CUDA mode compiles optimized (CPPJIT_OPT_LEVEL=0 opts out)"""

        import cppjit
        from support import OPT_LEVEL

        optimized = (
            cppjit.evaluate("""#ifdef __OPTIMIZE__
                            true
                            #else
                            false
                            #endif\n""")
            == 1
        )
        assert optimized == (OPT_LEVEL != 0)

    def test14_kernel_takes_device_interface(self):
        """JIT'd kernels take buffers offered through the array interface"""

        import cppjit
        from pytest import raises

        n = 128
        dev = cppjit.gbl.cppjit_cuda_iota(n)
        ptr = int(cppjit.addressof(dev))

        class Producer:
            def __init__(self, stream=None, typestr="<i4", readonly=False):
                self.__cuda_array_interface__ = {
                    "version": 3,
                    "shape": (n,),
                    "typestr": typestr,
                    "data": (ptr, readonly),
                    "strides": None,
                    "stream": stream,
                }

        cppjit.gbl.cppjit_cuda_scale[2, 64](Producer(), n, 3)
        # 1 is the legacy default stream, the one the launch already uses
        cppjit.gbl.cppjit_cuda_scale[2, 64](Producer(stream=1), n, 5)
        # a view is an exporter too, so it serves both launch surfaces
        import cppjit.cuda

        cppjit.gbl.cppjit_cuda_scale[2, 64](cppjit.cuda.view(Producer()), n, 2)
        total = cppjit.gbl.cppjit_cuda_sum_free(dev, n)
        assert total == 30 * n * (n - 1) // 2

        with raises(TypeError):  # 0 is not a stream in the protocol
            cppjit.gbl.cppjit_cuda_scale[2, 64](Producer(stream=0), n, 3)
        with raises(TypeError):  # no view type for that element type
            cppjit.gbl.cppjit_cuda_scale[2, 64](Producer(typestr="<m8"), n, 3)
        with raises(TypeError):  # host memory must not reach a kernel
            cppjit.gbl.cppjit_cuda_scale[2, 64](bytearray(4 * n), n, 3)

    def test16_kernel_rejects_unusable_buffers(self):
        """buffers a kernel cannot index densely are refused, not misread"""

        import cppjit
        from pytest import raises

        n = 32
        dev = cppjit.gbl.cppjit_cuda_iota(n)
        ptr = int(cppjit.addressof(dev))

        def producer(**over):
            spec = {
                "version": 3,
                "shape": (n,),
                "typestr": "<i4",
                "data": (ptr, False),
                "strides": None,
                "stream": None,
            }
            spec.update(over)
            return type("P", (), {"__cuda_array_interface__": spec})()

        # a kernel receives a pointer, not a layout: strided input would be
        # read as if it were dense
        with raises(TypeError, match="contiguous"):
            cppjit.gbl.cppjit_cuda_scale[1, 32](producer(strides=(8,)), n, 3)
        with raises(TypeError, match="version"):
            cppjit.gbl.cppjit_cuda_scale[1, 32](producer(version=4), n, 3)
        with raises(TypeError, match="mask"):
            cppjit.gbl.cppjit_cuda_scale[1, 32](producer(mask=object()), n, 3)
        with raises(TypeError, match="dict"):
            cppjit.gbl.cppjit_cuda_scale[1, 32](
                type("B", (), {"__cuda_array_interface__": [1, 2]})(), n, 3
            )

        # dense strides describe the same buffer and stay accepted
        cppjit.gbl.cppjit_cuda_scale[1, 32](producer(strides=(4,)), n, 3)
        assert cppjit.gbl.cppjit_cuda_sum_free(dev, n) == 3 * n * (n - 1) // 2

    def test17_read_only_buffers_need_const(self):
        """a read-only buffer binds to a const parameter, not a mutable one"""

        import cppjit
        from pytest import raises

        cppjit.cppdef("""
        __global__ void cppjit_cuda_copy(const int* v, int* out, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) out[i] = v[i];
        }
        """)

        n = 32
        src = cppjit.gbl.cppjit_cuda_iota(n)
        dst = cppjit.gbl.cppjit_cuda_iota(n)
        ptr = int(cppjit.addressof(src))
        read_only = type(
            "RO",
            (),
            {
                "__cuda_array_interface__": {
                    "version": 3,
                    "shape": (n,),
                    "typestr": "<i4",
                    "data": (ptr, True),
                    "strides": None,
                    "stream": None,
                }
            },
        )()

        with raises(TypeError, match="read-only"):
            cppjit.gbl.cppjit_cuda_scale[1, 32](read_only, n, 3)
        cppjit.gbl.cppjit_cuda_copy[1, 32](read_only, dst, n)
        assert cppjit.gbl.cppjit_cuda_sum_free(dst, n) == n * (n - 1) // 2
        cppjit.gbl.cppjit_cuda_sum_free(src, n)

    def test15_kernel_rejects_host_arrays(self):
        """numpy arrays are refused rather than passed as host pointers"""

        import cppjit
        import numpy as np
        from pytest import raises

        n = 64
        with raises(TypeError):
            cppjit.gbl.cppjit_cuda_scale[1, 64](np.arange(n, dtype=np.int32), n, 2)

        class DLPackOnly:
            def __dlpack_device__(self):
                return (2, 0)  # kDLCUDA

        with raises(TypeError):
            cppjit.gbl.cppjit_cuda_scale[1, 64](DLPackOnly(), n, 2)
