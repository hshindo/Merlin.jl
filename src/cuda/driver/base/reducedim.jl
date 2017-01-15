import Base: sum, maximum

macro reducedim(op, x, dim)
    quote
        op = $(esc(op))
        x = $(esc(x))
        dim = $(esc(dim))
        t = ctype(T)
        f = @nvrtc """
        $array_h
        template<typename T>
        __inline__ __device__ T warpReduce(T a) {
            for (int delta = warpSize/2; delta > 0; delta /= 2) {
                T b = __shfl_down(a, delta);
                a = $op;
            }
            return a;
        }

        template<typename T>
        __inline__ __device__ T blockReduce(T value) {
            static __shared__ T temp[32];
            int laneId = threadIdx.y % warpSize;
            int warpId = threadIdx.y / warpSize;

            value = warpReduce<T>(value);
            if (laneId == 0) temp[warpId] = value;
            __syncthreads();

            value = (laneId < 32) ? temp[laneId] : 0;
            value = warpReduce<T>(value);
            return value;
        }

        __global__ void reduce(Array<$t,3> x, Array<$t,3> y) {
            int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
            int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
            int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
            if (idx_y >= x.dims[1]) return;

            $t a = x(idx_x, idx_y, idx_z);
            for (int i = idx_y+blockDim.y*gridDim.y; i < x.dims[1]; i += blockDim.y*gridDim.y) {
                $t b = x(idx_x, i, idx_z);
                a = $op;
            }
            a = blockReduce<$t>(a);
            if (threadIdx.y == 0) y(blockIdx.x, blockIdx.y, blockIdx.z) = a;
        } """
        x3d = reshape3d(x, dim)
        by = 1024
        while true
            gy = Int(ceil(size(x3d,2)/by))
            y = similar(x, ntuple(i -> i==dim ? gy : size(x,i), ndims(x)))
            y3d = reshape3d(y, dim)
            f(x3d, y3d, dx=size(x3d,1), dy=size(x3d,2), dz=size(x3d,3), bx=1, by=by, bz=1)
            gy == 1 && return y
            x3d = y3d
        end
    end
end

maximum{T,N}(x::CuArray{T,N}, dim::Int) = @reducedim "a > b ? a : b" x dim
sum{T,N}(x::CuArray{T,N}, dim::Int) = @reducedim "a + b" x dim

function reshape3d(x::CuArray, dim::Int)
    dim1, dim2, dim3 = 1, size(x,dim), 1
    for i = 1:dim-1
        dim1 *= size(x,i)
    end
    for i = dim+1:ndims(x)
        dim3 *= size(x,i)
    end
    reshape(x, dim1, dim2, dim3)
end
