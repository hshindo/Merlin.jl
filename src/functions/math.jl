import Base: +, -, *, .*

+(x1::Var, x2::Var) = axsum([1.0,1.0], [x1,x2])
+(a::Number, x::Var) = Var(a) + x
+(x::Var, a::Number) = x + Var(a)

-(x1::Var, x2::Var) = axsum([1.0,-1.0], [x1,x2])
-(a::Number, x::Var) = Var(a) - x
-(x::Var, a::Number) = x - Var(a)
-(x::Var) = axsum([-1.0], [x])

*(a::Number, x::Var) = axsum([a], [x])
*(x::Var, a::Number) = a * x

"""
    axsum

y = as[1] * xs[1] + as[2] * xs[2] + ...
"""
function axsum(as::Vector{Float64}, xs::Vector{Var})
    maxi, maxlen = 0, 0
    for i = 1:length(xs)
        n = length(xs[i].data)
        n <= maxlen && continue
        maxi = i
        maxlen = n
    end
    y = zeros(xs[maxi].data)
    for i = 1:length(xs)
        y = axpy!(as[i], xs[i].data, y)
    end

    function df(gy)
        for i = 1:length(xs)
            a, x = as[i], xs[i]
            hasgrad(x) && (x.grad = ∇axpy!(a,x.grad,gy))
        end
    end
    Var(y, xs, axsum, df)
end

function axpy!{T}(a::Float64, x::Array{T}, y::Array{T})
    n = length(x)
    for k = 1:n:length(y)
        BLAS.axpy!(n, T(a), pointer(x), 1, pointer(y,k), 1)
    end
    y
end
axpy!(a::Float64, x::Number, y::Array) = broadcast!(+, y, y, a*x)
axpy!(a::Float64, x::Number, y::Number) = a * x + y

function ∇axpy!{T}(a::Float64, gx::Array{T}, gy::Array{T})
    n = length(gx)
    for k = 1:n:length(gy)
        BLAS.axpy!(n, T(a), pointer(gy,k), 1, pointer(gx), 1)
    end
    gx
end
∇axpy!(a::Float64, gx::Number, gy::Array) = gx + a * sum(gy)
∇axpy!(a::Float64, gx::Number, gy::Number) = gx + a * gy

function .*(x1::Var, x2::Var)
    y = x1.data .* x2.data
    function df(gy)
        hasgrad(x1) && ∇elemtimes!(x2.data, x1.grad, gy)
        hasgrad(x2) && ∇elemtimes!(x1.data, x2.grad, gy)
    end
    Var(y, [x1,x2], .*, df)
end

function *(x1::Var, x2::Var)
    y = x1.data * x2.data
    function df(gy)
        T = eltype(gy)
        hasgrad(x1) && BLAS.gemm!('N', 'T', T(1), gy, x2.data, T(1), x1.grad)
        hasgrad(x2) && BLAS.gemm!('T', 'N', T(1), x1.data, gy, T(1), x2.grad)
    end
    Var(y, [x1,x2], *, df)
end

function ∇elemtimes!{T,N}(x2::Array{T,N}, gx1::Array{T,N}, gy::Array{T,N})
    if length(gx1) < length(gy)
        @inbounds for k = 1:length(gx1):length(gy)
            @simd for i = 1:length(gx1)
                gx1[i] += gy[k+i-1] * x2[k+i-1]
            end
        end
    else
        broadcast!(.+, gx1, gx1, gy.*x2)
    end
end

import Base: transpose

"""
    transpose(x::Var)
"""
function transpose(x::Var)
    y = transpose(x.data)
    df{T}(gy::UniArray{T}) = hasgrad(x) && BLAS.axpy!(T(1), transpose(gy), x.grad)
    Var(y, [x], transpose, df)
end
