import Base: exp, log, transpose
import Base: +, .+, -, .-, .*, *

"""
    exp(x::Var)
"""
function exp(x::Var)
    y = Var(exp(x.data), exp, (x,))
    y.df! = function df!()
        isvoid(x.grad) || ∇exp!(y.data, y.grad, x.grad)
    end
    y
end

function ∇exp!{T}(y::Array{T}, gy::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * y[i]
    end
end

@generated function ∇exp!{T}(y::CuArray{T}, gy::CuArray{T}, gx::CuArray{T})
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

"""
    log(x::Var)
"""
function log(x::Var)
    y = Var(log(x.data), log, (x,))
    y.df! = function df!()
        isvoid(x.grad) || ∇log!(y.grad, x.data, x.grad)
    end
    y
end

function ∇log!{T}(gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] / x[i]
    end
end

@generated function ∇log!{T}(gy::CuArray{T}, x::CuArray{T}, gx::CuArray{T})
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

"""
    transpose(x::Var)
"""
function transpose(x::Var)
    y = Var(transpose(x.data), transpose, (x,))
    y.df! = function df!()
        isvoid(x.grad) || BLAS.axpy!(eltype(x.data)(1), transpose(y.grad), x.grad)
    end
    y
end

"""
    +(x1::Var, x2::Var)
"""
function +(x1::Var, x2::Var)
    y = Var(x1.data + x2.data, +, (x1,x2))
    y.df! = function df!()
        T = eltype(y.grad)
        isvoid(x1.grad) || BLAS.axpy!(T(1), y.grad, x1.grad)
        isvoid(x2.grad) || BLAS.axpy!(T(1), y.grad, x2.grad)
    end
    y
end
+(a::Number, x::Var) = Var(a) + x
+(x::Var, a::Number) = x + Var(a)

"""
    .+(x1::Var, x2::Var)
"""
function .+(x1::Var, x2::Var)
    y = Var(x1.data .+ x2.data, .+, (x1,x2))
    y.df! = function df!()
        isvoid(x1.grad) || ∇elemplus!(y.grad, x1.grad)
        isvoid(x2.grad) || ∇elemplus!(y.grad, x2.grad)
    end
    y
end

function ∇elemplus!{T,N}(gy::UniArray{T,N}, gx::UniArray{T,N})
    for i = 1:N
        size(gx,i) == 1 && size(gy,i) > 1 && (gy = sum(gy,i))
    end
    BLAS.axpy!(T(1), gy, gx)
end
∇elemplus!{T,N}(gy::Array{T,N}, gx::Array{T}) = ∇elemplus!(gy, redim(gx,N))

"""
    -(x1::Var, x2::Var)
    -(x::Var)
"""
function -(x1::Var, x2::Var)
    y = Var(x1.data - x2.data, -, (x1,x2))
    y.df! = function df!()
        T = eltype(y.grad)
        isvoid(x1.grad) || BLAS.axpy!(T(1), y.grad, x1.grad)
        isvoid(x2.grad) || BLAS.axpy!(T(-1), y.grad, x2.grad)
    end
    y
end

function -(x::Var)
    y = Var(-x.data, -, (x,))
    y.df! = function df!()
        T = eltype(y.grad)
        isvoid(x.grad) || BLAS.axpy!(T(-1), y.grad, x.grad)
    end
    y
end
-(a::Number, x::Var) = Var(a) - x
-(x::Var, a::Number) = x - Var(a)

"""
    .-(x1::Var, x2::Var)
"""
function .-(x1::Var, x2::Var)
    y = Var(x1.data .- x2.data, .-, (x1,x2))
    y.df! = function df!()
        isvoid(x1.grad) || ∇elemplus!(y.grad, x1.grad)
        isvoid(x2.grad) || ∇elemminus!(y.grad, x2.grad)
    end
    y
end

function ∇elemminus!{T,N}(gy::UniArray{T,N}, gx::UniArray{T,N})
    for i = 1:N
        size(gx,i) == 1 && size(gy,i) > 1 && (gy = sum(gy,i))
    end
    BLAS.axpy!(T(-1), gy, gx)
end

#=
function ∇elemminus!{T}(gy::Array{T}, gx::Array{T})
    ind_gx = CartesianIndex(size(gx))
    @inbounds @simd for I in CartesianRange(size(gy))
        gx[min(ind_gx,I)] -= gy[I]
    end
end
=#

"""
    \.\*(x1::Var, x2::Var)
"""
function .*(x1::Var, x2::Var)
    y = Var(x1.data .* x2.data, .*, (x1,x2))
    y.df! = function df!()
        isvoid(x1.grad) || ∇elemtimes!(y.grad, x2.data, x1.grad)
        isvoid(x2.grad) || ∇elemtimes!(y.grad, x1.data, x2.grad)
    end
    y
end

function ∇elemtimes!{T,N}(gy::Array{T,N}, x2::Array{T,N}, gx1::Array{T,N})
    if size(x2) == size(gx1)
        @inbounds @simd for i = 1:length(gy)
            gx1[i] += gy[i] * x2[i]
        end
    else
        gx = gy .* x2
        for i = 1:N
            size(gx1,i) == 1 && size(gx,i) > 1 && (gx = sum(gx,i))
        end
        BLAS.axpy!(T(1), gx, gx1)
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

#=
function ∇elemtimes!{T}(gy::Array{T}, x2::Array{T}, gx1::Array{T})
    ind_x2 = CartesianIndex(size(x2))
    ind_gx1 = CartesianIndex(size(gx1))
    @inbounds @simd for I in CartesianRange(size(gy))
        gx1[min(ind_gx1,I)] += gy[I] * x2[min(ind_x2,I)]
    end
end
=#

"""
    \*(x1::Var, x2::Var)
"""
function *(x1::Var, x2::Var)
    y = ndims(x2.data) == 1 ? gemv(x1,x2) : gemm(x1,x2)
    y.f = *
    y
end
