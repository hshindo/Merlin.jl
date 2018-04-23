import Base: reducedim, sum, maximum
export argmax, argmin

function reshape3d(x::CuArray, dim::Int)
    # dim == 0 && return (1, length(x), 1)
    dim1, dim2, dim3 = 1, size(x,dim), 1
    for i = 1:dim-1
        dim1 *= size(x, i)
    end
    for i = dim+1:ndims(x)
        dim3 *= size(x, i)
    end
    reshape(x, dim1, dim2, dim3)
end


function reducedim_f(T, v0, op)
    CuFunction("""
    template<typename T>
    __inline__ __device__ void warpReduce(T &a) {
        for (int delta = warpSize/2; delta > 0; delta /= 2) {
            T b = __shfl_down(a, delta);
            a = $op;
        }
    }

    template<typename T>
    __inline__ __device__ void blockReduce(T &val) {
        static __shared__ T shared[32]; // 32*32=1024
        int laneId = threadIdx.x % warpSize;
        int warpId = threadIdx.x / warpSize;
        warpReduce<T>(val);
        if (laneId == 0) shared[warpId] = val;
        __syncthreads();

        val = shared[laneId];
        if (warpId == 0) warpReduce<T>(val);
    }

    __global__ void reduce($T *x, $T *y, int size1, int size2) {
        int idxX = blockIdx.x * blockDim.x + threadIdx.x;
        int idxY = blockIdx.y * blockDim.y + threadIdx.y;
        if (idxX >= size1) return;
        if (idxY >= size2) return;

        $T val = x[idxX + idxY * size2];
        blockReduce<$T>(val);
        if (idxX == 0) y[idxY] = val;
        //if (threadIdx.x == 0) y[blockIdx.x] = val;
    }
    """)
end

@generated function sum2(x::CuArray{T}, dim::Int) where T
    f = reducedim_f(cstring(T), "0", "a + b")
    quote
        #x3d = reshape3d(x, dim)
        #xsize = [size(x)...]
        #xsize[dim] = 1
        #y = similar(x, xsize...)

        y = similar(x, 1, size(x,2))
        #n = size(x,2)
        bx = 1024
        #gx = n <= bx ? 1 : ceil(Int, n/bx)
        gdims, bdims = (1,size(x,2),1), (bx,1,1)
        # gdims, bdims = cudims(length(x))
        culaunch($f, gdims, bdims, x.ptr, y.ptr, size(x,1), size(x,2))
        y
    end
end

function find_f{T}(::Type{T}, v0::String, op::String)
    CuFunction("""
    template<typename T>
    __inline__ __device__ void warpReduce(T &a, int &i) {
        for (int delta = warpSize/2; delta > 0; delta /= 2) {
            T b = __shfl_down(a, delta);
            int j = __shfl_down(i, delta);
            if ($op) {
                a = b;
                i = j;
            }
        }
    }
    template<typename T>
    __inline__ __device__ void blockReduce(T &value, int &index) {
        static __shared__ T values[32]; // 32*32=1024
        static __shared__ int indexs[32];
        int laneId = threadIdx.y % warpSize;
        int warpId = threadIdx.y / warpSize;
        warpReduce<T>(value, index);
        if (laneId == 0) {
            values[warpId] = value;
            indexs[warpId] = index;
        }
        __syncthreads();
        value = (laneId < 32) ? values[laneId] : $v0;
        index = (laneId < 32) ? indexs[laneId] : -1;
        warpReduce<T>(value, index);
    }
    __global__ void f(Array<$T,3> x, Array<int,3> y) {
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
        $T value = (idx_y < x.dims[1]) ? x(idx_x,idx_y,idx_z) : $v0;
        int index = (idx_y < x.dims[1]) ? blockIdx.y * blockDim.y + threadIdx.y : -1;
        //for (int i = idx_y+blockDim.y*gridDim.y; i < x.dims[1]; i += blockDim.y*gridDim.y) {
        //    $T b = x(idx_x, i, idx_z);
        //    a = $op;
        //}
        blockReduce<$T>(value, index);
        if (threadIdx.y == 0) y(blockIdx.x, blockIdx.y, blockIdx.z) = index + 1;
    }
    """)
end

function reducedim_f2(T, op::String)
    blocksize = 1024
    CuFunction("""
    __global__ void reduce($T *x, $T *y) {
        static __shared__ $T temp[1024];
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        //int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
        //if (idx_y >= x.dims[1]) return;
        int tid = threadIdx.y;
        int idx_y = blockIdx.y * $blocksize * 2 + threadIdx.y;
        $T a = (idx_y < x.dims[1]) ? x(idx_x, idx_y, idx_z) : 0;

        if (idx_y+$blocksize < x.dims[1]) {
            a += x[idx_y+$blocksize];
        }
        temp[tid] = a;
        $T b;
        __syncthreads();

        if (($blocksize >= 512) && (tid < 256)) {
            b = temp[tid+256];
            a = $op;
            temp[tid] = a;
            __syncthreads();
        }
        if (($blocksize >= 256) &&(tid < 128)) {
            b = temp[tid+128];
            a = $op;
            temp[tid] = a;
            __syncthreads();
        }
        if (($blocksize >= 128) && (tid < 64)) {
            b = temp[tid+64];
            a = $op;
            temp[tid] = a;
            __syncthreads();
        }
        if (tid < 32) {
            if ($blocksize >= 64) a += temp[tid+32];
            for (int offset = warpSize/2; offset > 0; offset /= 2) {
                a += __shfl_down(a, offset);
            }
        }
        if (threadIdx.y == 0) y(blockIdx.x, blockIdx.y, blockIdx.z) = a;
    }""")
end
