import Base: exp, log
import Base: .+, +, .-, -, .*, *

macro elemwise(op)
    quote
        op = $op
        y = CuArray{T}(size(x))
        t = ctype(T)
        f = @nvrtc """
        $array_h
        __global__ void f(Array<$t,$N> x, Array<$t,$N> y) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < y.length()) {
                y(idx) = $op(x(idx));
            }
        } """
        f(x, y, dx=length(y))
        y
    end
end

for op in (:exp, :log)
    @eval begin
        function $op{T,N}(x::AbstractCuArray{T,N})
            op = $op
            y = CuArray{T}(size(x))
            t = ctype(T)
            f = @nvrtc """
            $array_h
            __global__ void f(Array<$t,$N> x, Array<$t,$N> y) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < y.length()) {
                    y(idx) = $op(x(idx));
                }
            } """
            f(x, y, dx=length(y))
            y
        end
    end
end

for op in (:+, :-)
    @eval begin
        function $op{T,N}(x1::AbstractCuArray{T,N}, x2::AbstractCuArray{T,N})
            size(x1) == size(x2) || throw(DimensionMismatch())
            op = $op
            y = CuArray{T}(size(x1))
            t = ctype(T)
            f = @nvrtc """
            $array_h
            __global__ void f(Array<$t,$N> x1, Array<$t,$N> x2, Array<$t,$N> y) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < y.length()) {
                    y(idx) = x1(idx) $op x2(idx);
                }
            } """
            f(x1, x2, y, dx=length(y))
            y
        end
    end
end

for (op1,op2) in ((:.+,:+), (:.-,:-))
    @eval begin
        function $op1{T,N}(x1::AbstractCuArray{T,N}, x2::AbstractCuArray{T,N})
            dims = ntuple(i -> max(size(x1,i),size(x2,i)), N)
            y = CuArray{T}(dims)
            broadcast!($op2, y, x1, x2)
        end
    end
end

*(x1::CuMatrix, x2::CuMatrix) = BLAS.gemm(x1, x2)
