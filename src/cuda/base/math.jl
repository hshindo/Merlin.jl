import Base: exp, log
import Base: .+, +, .-, -, .*, *

for op in (:exp, :log)
    @eval begin
        @generated function $op{T}(x::CuArray{T})
            op = $op
            f = CuFunction("""
            __global__ void f($T *x, $T *y, int length) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < length) {
                    y[idx] = $op(x[idx]);
                }
            }""")
            quote
                y = similar(x)
                $f(x.ptr, y.ptr, length(x), dx=length(x))
                y
            end
        end
    end
end

for op in (:+, :-)
    @eval begin
        @generated function $op{T}(x1::CuArray{T}, x2::CuArray{T})
            size(x1) == size(x2) || throw(DimensionMismatch())
            op = $op
            f = CuFunction("""
            __global__ void f($T *x1, $T *x2, $T *y, int length) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < length) {
                    y[idx] = x1[idx] $op x2[idx];
                }
            }""")
            quote
                y = similar(x1)
                $f(x1.ptr, x2.ptr, y.ptr, length(y), dx=length(y))
                y
            end
        end
    end
end

#=
for op in (:.+, :.-, :.*)
    @eval begin
        @generated function $op{T}(x1::CuArray{T}, x2::CuArray{T})
            quote
                dims = ntuple(i -> max(size(x1,i),size(x2,i)), N)
                y = CuArray{T}(dims)
                broadcast!($op2, y, x1, x2)
            end
        end
    end
end
=#
