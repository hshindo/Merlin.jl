@generated function ∇exp!(y::CuArray{T}, gy::CuArray{T}, gx::CuArray{T}) where T
    f = CuFunction("""
    __global__ void f($T *y, $T *gy, $T *gx, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            gx[idx] += gy[idx] * y[idx];
        }
    }""")
    quote
        $f(y.ptr, gy.ptr, gx.ptr, length(y), dx=length(y))
    end
end

@generated function ∇log!(gy::CuArray{T}, x::CuArray{T}, gx::CuArray{T}) where T
    f = CuFunction("""
    __global__ void f($T *gy, $T *x, $T *gx, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            gx[idx] += gy[idx] / x[idx];
        }
    }""")
    quote
        $f(gy.ptr, x.ptr, gx.ptr, length(gy), dx=length(gy))
    end
end

@generated function ∇elemtimes!{T,N}(gy::CuArray{T,N}, x2::CuArray{T,N}, gx1::CuArray{T,N})
    f = CuFunction("""
    __global__ void f(Array<$T,$N> gy, Array<$T,$N> x2, Array<$T,$N> gx1) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < gy.length()) {
            gx1[idx] += gy[idx] * x2[idx];
        }
    }""")
    quote
        if size(x2) == size(gx1)
            $f(gy, x2, gx1, dx=length(gy))
        else
            gx = gy .* x2
            for i = 1:N
                size(gx1,i) == 1 && size(gx,i) > 1 && (gx = sum(gx,i))
            end
            BLAS.axpy!(T(1), gx, gx1)
        end
    end
end

function ∇elemtimes!{T}(gy::Array{T}, x2::Array{T}, gx1::Array{T})
    ind_x2 = CartesianIndex(size(x2))
    ind_gx1 = CartesianIndex(size(gx1))
    @inbounds @simd for I in CartesianRange(size(gy))
        gx1[min(ind_gx1,I)] += gy[I] * x2[min(ind_x2,I)]
    end
end
