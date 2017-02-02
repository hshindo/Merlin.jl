import Base: exp, log, transpose
import Base: +, .+, -, .-, .*, *

"""
    exp(x::Var)
"""
exp(x::Var) = forward(exp, x)

function forward(::typeof(exp), x::Array)
    y = exp(x)
    backward!(gy, gx) = isvoid(gx) || ∇exp!(y, gy, gx)
    y, backward!
end

function ∇exp!{T}(y::Array{T}, gy::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * y[i]
    end
end

"""
    log(x::Var)
"""
log(x::Var) = forward(log, x)

function forward(::typeof(log), x::Array)
    y = log(x)
    backward!(gy, gx) = isvoid(gx) || ∇log!(gy, x, gx)
    y, backward!
end

function ∇log!{T}(gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] / x[i]
    end
end

"""
    transpose(x::Var)
"""
transpose(x::Var) = forward(transpose, x)

function forward{T}(::typeof(transpose), x::Array{T})
    y = transpose(x)
    backward!(gy, gx) = isvoid(gx) || BLAS.axpy!(T(1), transpose(gy), gx)
    y, backward!
end

"""
    +(x1::Var, x2::Var)
"""
+(x1::Var, x2::Var) = forward(+, x1, x2)
+(a::Number, x::Var) = Var([a]) + x
+(x::Var, a::Number) = x + Var([a])

function forward(::typeof(+), x1::Array, x2::Array)
    y = x1 + x2
    function backward!(gy, gx1, gx2)
        isvoid(gx1) || add!(gx1, gy)
        isvoid(gx2) || add!(gx2, gy)
    end
    y, backward!
end

"""
    .+(x1::Var, x2::Var)
"""
.+(x1::Var, x2::Var) = forward(.+, x1, x2)

function forward(::typeof(.+), x1::Array, x2::Array)
    y = x1 .+ x2
    function backward!(gy, gx1, gx2)
        isvoid(gx1) || ∇elemplus!(gy, gx1)
        isvoid(gx2) || ∇elemplus!(gy, gx2)
    end
    y, backward!
end

function ∇elemplus!{T}(gy::Array{T}, gx::Array{T})
    ind_gx = CartesianIndex(size(gx))
    @inbounds @simd for I in CartesianRange(size(gy))
        gx[min(ind_gx,I)] += gy[I]
    end
end

"""
    -(x1::Var, x2::Var)
    -(x::Var)
"""
-(x1::Var, x2::Var) = forward(-, x1, x2)
-(a::Number, x::Var) = Var([a]) - x
-(x::Var, a::Number) = x - Var([a])
-(x::Var) = forward(-, x)

function forward(::typeof(-), x1::Array, x2::Array)
    y = x1 - x2
    function backward!(gy, gx1, gx2)
        isvoid(gx1) || add!(gx1, gy)
        isvoid(gx2) || BLAS.axpy!(eltype(gy)(-1), gy, gx2)
    end
    y, backward!
end

function forward(::typeof(-), x::Array)
    y = -x
    backward!(gy, gx) = isvoid(gx) || BLAS.axpy!(eltype(gy)(-1), gy, gx)
    y, backward!
end

"""
    .-(x1::Var, x2::Var)
"""
.-(x1::Var, x2::Var) = forward(.-, x1, x2)

function forward(::typeof(.-), x1::Array, x2::Array)
    y = x1 .- x2
    function backward!(gy, gx1, gx2)
        isvoid(gx1) || ∇elemplus!(gy, gx1)
        isvoid(gx2) || ∇elemminus!(gy, gx2)
    end
    y, backward!
end

function ∇elemminus!{T}(gy::Array{T}, gx::Array{T})
    ind_gx = CartesianIndex(size(gx))
    @inbounds @simd for I in CartesianRange(size(gy))
        gx[min(ind_gx,I)] -= gy[I]
    end
end

"""
    \.\*(x1::Var, x2::Var)
"""
.*(x1::Var, x2::Var) = forward(.*, x1, x2)

function forward(::typeof(.*), x1::Array, x2::Array)
    y = x1 .* x2
    function backward!(gy, gx1, gx2)
        isvoid(gx1) || ∇elemtimes!(gy, x2, gx1)
        isvoid(gx2) || ∇elemtimes!(gy, x1, gx2)
    end
    y, backward!
end

function ∇elemtimes!{T}(gy::Array{T}, x2::Array{T}, gx1::Array{T})
    ind_x2 = CartesianIndex(size(x2))
    ind_gx1 = CartesianIndex(size(gx1))
    @inbounds @simd for I in CartesianRange(size(gy))
        gx1[min(ind_gx1,I)] += gy[I] * x2[min(ind_x2,I)]
    end
end

"""
    \*(x1::Var, x2::Var)
"""
*(x1::Var, x2::Var) = forward(*, x1, x2)

forward(::typeof(*), x1::Matrix, x2::Vector) = forward(gemv, 'N', 1, x1, x2)
forward(::typeof(*), x1::Matrix, x2::Matrix) = forward(gemm, 'N', 'N', 1, x1, x2)
