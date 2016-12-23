import Base: exp, log
import Base: +, -, *, .*

"""
    exp
"""
function exp(x::Var)
    y = exp(x.data)
    df(gy) = isvoid(x.grad) || ∇exp!(y, gy, x.grad)
    Var(y, df, (x,))
end
exp(x::Var{Void}) = Var(Void(), exp, (x,))

function ∇exp!{T}(y::Array{T}, gy::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * y[i]
    end
end

"""
    log
"""
function log(x::Var)
    y = log(x.data)
    df(gy) = isvoid(x.grad) || ∇log!(gy, x.data, x.grad)
    Var(y, df, (x,))
end
log(x::Var{Void}) = Var(Void(), log, (x,))

function ∇log!{T}(gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] / x[i]
    end
end

"""
    +(x1::Var, x2::Var)
"""
function +(x1::Var, x2::Var)
    (isvoid(x1.data) || isvoid(x2.data)) && return Var(Void(), +, (x1,x2))
    y = x1.data + x2.data
    function df(gy)
        isvoid(x1.grad) || broadcast!(+, x1.grad, x1.grad, gy)
        isvoid(x2.grad) || broadcast!(+, x2.grad, x2.grad, gy)
    end
    Var(y, df, (x1,x2))
end
+(a::Number, x::Var) = Var([a]) + x
+(x::Var, a::Number) = x + Var([a])

"""
    -(x1::Var, x2::Var)
    -(x::Var)

Automatically broadcasted.
"""
function -(x1::Var, x2::Var)
    (isvoid(x1.data) || isvoid(x2.data)) && return Var(Void(), -, (x1,x2))
    y = x1.data - x2.data
    df(gy) = begin
        isvoid(x1.grad) || broadcast!(+, x1.grad, x1.grad, gy)
        isvoid(x2.grad) || broadcast!(-, x2.grad, x2.grad, gy)
    end
    Var(y, df, (x1,x2))
end
-(a::Number, x::Var) = Var([a]) - x
-(x::Var, a::Number) = x - Var([a])

function -(x::Var)
    y = -x.data
    df(gy) = isvoid(x.grad) || broadcast!(-, x.grad, x.grad, gy)
    Var(y, df, (x,))
end
-(x::Var{Void}) = Var(Void(), -, (x,))

mat(x::Vector) = reshape(x,length(x),1)
mat(x::Matrix) = x

"""
    \*(x1::Var, x2::Var)
"""
function *(x1::Var, x2::Var)
    (isvoid(x1.data) || isvoid(x2.data)) && return Var(Void(), *, (x1,x2))
    ndims(x2.data) == 1 && return gemv(x1, x2)
    ndims(x2.data) == 2 && size(x2.data,2) == 1 && return gemv(x1, Var(x2,data=vec(x2.data)))
    gemm(x1, x2)
end

"""
    \.\*(x1::Var, x2::Var)
"""
function .*(x1::Var, x2::Var)
    (isvoid(x1.data) || isvoid(x2.data)) && return Var(Void(), .*, (x1,x2))
    length(x1) == length(x2) || throw(DimensionMismatch())
    y = x1.data .* x2.data
    function df(gy)
        isvoid(x1.grad) || ∇elemtimes!(gy, x2.data, x1.grad)
        isvoid(x2.grad) || ∇elemtimes!(gy, x1.data, x2.grad)
    end
    Var(y, df, (x1,x2))
end

function ∇elemtimes!{T}(gy::Array{T}, x2::Array{T}, gx1::Array{T})
    @inbounds @simd for i = 1:length(gy)
        gx1[i] += gy[i] * x2[i]
    end
end

function ∇elemtimes2!(x2, gx1, gy)
    if length(gx1) < length(gy)
        @inbounds for k = 0:length(gx1):length(gy)-1
            @simd for i = 1:length(gx1)
                gx1[i] += gy[i+k] * x2[i+k]
            end
        end
    else
        @inbounds for k = 0:length(x2):length(gy)-1
            @simd for i = 1:length(x2)
                gx1[i+k] += gy[i+k] * x2[i]
            end
        end
    end
end
