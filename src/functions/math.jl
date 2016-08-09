import Base: +, -, *

+(x1::Var, x2::Var) = wsum([1.0,1.0], [x1,x2])
+(a::Number, x::Var) = Var(a) + x
+(x::Var, a::Number) = x + Var(a)

-(x1::Var, x2::Var) = wsum([1.0,-1.0], [x1,x2])
-(a::Number, x::Var) = Var(a) - x
-(x::Var, a::Number) = x - Var(a)
-(x::Var) = wsum([-1.0], [x])

*(a::Number, x::Var) = wsum([a], [x])
*(x::Var, a::Number) = a * x

"""
y = as[1] * xs[1] + as[2] * xs[2] + ...
"""
function wsum(as::Vector{Float64}, xs::Vector{Var})
    @assert length(as) == length(xs)
    maxi, maxlen = 0, 0
    for i = 1:length(xs)
        n = length(xs[i].data)
        n <= maxlen && continue
        maxi = i
        maxlen = n
    end
    y = zeros(xs[maxi].data)
    T = eltype(y)
    for i = 1:length(xs)
        add!(as[i], xs[i].data, y)
    end
    function df(gy)
        for i = 1:length(xs)
            hasgrad(xs[i]) && ∇add!(as[i], xs[i].grad, gy)
        end
    end
    Var(y, xs, wsum, df)
end

function add!{T}(a::Number, x::Array{T}, y::Array{T})
    n = length(x)
    for k = 1:n:length(y)
        BLAS.axpy!(n, T(a), pointer(x), 1, pointer(y,k), 1)
    end
end
add!{T}(a::Number, x::Number, y::Array{T}) = broadcast!(+, y, y, x)

function ∇add!{T}(a::Number, gx::Array{T}, gy::Array{T})
    n = length(gx)
    for k = 1:n:length(gy)
        BLAS.axpy!(n, T(a), pointer(gy,k), 1, pointer(gx), 1)
    end
end

import Base: .*, *

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
