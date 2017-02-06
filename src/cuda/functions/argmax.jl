import ..Merlin: argmax

function argmax{T,N}(x::CuArray{T,N}, dim::Int)
    t = ctype(T)
    f = @compile """
    $array_h
    template<typename T>
    __inline__ __device__ T warpReduce(T a, int *maxId) {
        for (int delta = warpSize/2; delta > 0; delta /= 2) {
            T b = __shfl_down(a, delta);
            int id = __shfl_down(*maxId, delta);
            if (a < b) {
                a = b;
                *maxId = id;
            }
        }
        return a;
    }

    template<typename T>
    __inline__ __device__ int blockReduce(T value) {
        static __shared__ T shared[32];
        static __shared__ int maxIds[32];
        int laneId = threadIdx.y % warpSize;
        int warpId = threadIdx.y / warpSize;

        int maxId = blockIdx.y * blockDim.y + threadIdx.y;
        value = warpReduce<T>(value, &maxId);
        if (laneId == 0) {
            shared[warpId] = value;
            maxIds[warpId] = maxId;
        }
        __syncthreads();

        value = (laneId < 32) ? shared[laneId] : 0;
        maxId = (laneId < 32) ? maxIds[laneId] : 0;
        value = warpReduce<T>(value, &maxId);
        return maxId;
    }

    __global__ void f(Array<$t,3> x, Array<int,3> y) {
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
        if (idx_y >= x.dims[1]) return;

        $t value = x(idx_x, idx_y, idx_z);
        int maxId = blockReduce<$t>(value);
        if (threadIdx.y == 0) y(blockIdx.x, 0, blockIdx.z) = maxId + 1;
    } """
    y = CuArray{Cint}(ntuple(i -> i==dim ? 1 : size(x,i), N))
    x = reshape3d(x, dim)
    f(x, reshape3d(y,dim), dx=size(x,1), dy=size(x,2), dz=size(x,3), bx=1, by=1024, bz=1)
    y
end
