import Base: exp, log, transpose
import Base: +, .+, -, .-, .*, *

"""
    exp(x::Var)
"""
exp(x::Var) = forward0(exp, x)

function forward(::typeof(exp), x::UniArray)
    y = exp(x)
    backward!(gy, gx) = isvoid(gx) || ∇exp!(y, gy, gx)
    y, backward!
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
log(x::Var) = forward0(log, x)

function forward(::typeof(log), x::UniArray)
    y = log(x)
    backward!(gy, gx) = isvoid(gx) || ∇log!(y, x, gx)
    y, backward!
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
transpose(x::Var) = forward0(transpose, x)

function forward{T}(::typeof(transpose), x::UniArray{T})
    y = transpose(x)
    backward!(gy, gx) = isvoid(gx) || BLAS.axpy!(T(1), transpose(gy), gx)
    y, backward!
end

"""
    +(x1::Var, x2::Var)
"""
+(x1::Var, x2::Var) = forward0(+, x1, x2)

function forward{T}(::typeof(+), x1::UniArray{T}, x2::UniArray{T})
    y = x1 + x2
    function backward!(gy, gx1, gx2)
        isvoid(gx1) || BLAS.axpy!(T(1), gy, gx1)
        isvoid(gx2) || BLAS.axpy!(T(1), gy, gx2)
    end
    y, backward!
end
+(a::Number, x::Var) = Var(a) + x
+(x::Var, a::Number) = x + Var(a)

"""
    .+(x1::Var, x2::Var)
"""
.+(x1::Var, x2::Var) = forward0(+, x1, x2)

function forward(::typeof(.+), x1::UniArray, x2::UniArray)
    y = x1 .+ x2
    function backward!(gy, gx1, gx2)
        isvoid(gx1) || ∇elemplus!(gy, gx1)
        isvoid(gx2) || ∇elemplus!(gy, gx2)
    end
    y, backward!
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
-(x1::Var, x2::Var) = forward0(-, x1, x2)
-(x::Var) = forward0(-, x)

function forward{T}(::typeof(-), x1::UniArray{T}, x2::UniArray{T})
    y = x1 - x2
    function backward!(gy, gx1, gx2)
        isvoid(gx1) || BLAS.axpy!(T(1), gy, gx1)
        isvoid(gx2) || BLAS.axpy!(T(-1), gy, gx2)
    end
    y, backward!
end

function forward{T}(::typeof(-), x::UniArray{T})
    y = -x
    backward!(gy, gx) = isvoid(gx) || BLAS.axpy!(T(-1), gy, gx)
    y, backward!
end

-(a::Number, x::Var) = Var(a) - x
-(x::Var, a::Number) = x - Var(a)

"""
    .-(x1::Var, x2::Var)
"""
.-(x1::Var, x2::Var) = forward0(.-, x1, x2)

function forward(::typeof(.-), x1::UniArray, x2::UniArray)
    y = x1 .- x2
    function backward!(gy, gx1, gx2)
        isvoid(gx1) || ∇elemplus!(gy, gx1)
        isvoid(gx2) || ∇elemminus!(gy, gx2)
    end
    y, backward!
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
.*(x1::Var, x2::Var) = forward0(.*, x1, x2)

function forward(::typeof(.*), x1::UniArray, x2::UniArray)
    y = x1 .* x2
    function backward!(gy, gx1, gx2)
        isvoid(gx1) || ∇elemtimes!(gy, x2, gx1)
        isvoid(gx2) || ∇elemtimes!(gy, x1, gx2)
    end
    y, backward!
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
    ndims(x2.data) == 1 ? gemv(x1,x2) : gemm(x1, x2)
end
