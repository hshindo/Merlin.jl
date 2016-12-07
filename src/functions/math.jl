import Base: +, -, *, .*

"""
    +(x1::Var, x2::Var)

```julia
x1 = Var(rand(Float32,5,4))
x2 = Var(rand(Float32,5,4))
y = x1 + x2
```
"""
function +(x1::Var, x2::Var)
    (x1.data == nothing || x2.data == nothing) && return Var(nothing, +, (x1,x2))
    y = x1.data + x2.data
    function df(gy)
        isconst(x1) || broadcast!(x1.grad, x1.grad, gy)
        isconst(x2) || broadcast!(x2.grad, x2.grad, gy)
    end
    Var(y, +, (x1,x2), df)
end
+(a::Number, x::Var) = Var(a) + x
+(x::Var, a::Number) = x + Var(a)

"""
    -(x1::Var, x2::Var)
    -(x::Var)
"""
function -(x1::Var, x2::Var)
    (x1.data == nothing || x2.data == nothing) && return Var(nothing, -, (x1,x2))
    y = x1.data - x2.data
    df(gy) = begin
        isconst(x1) || (x1.grad .+= gy)
        isconst(x2) || (x2.grad .-= gy)
    end
    Var(y, -, (x1,x2), df)
end
#-(x::Var) = Var() - x

"""
    \*(x1::Var, x2::Var)
"""
*(x1::Var, x2::Var) = gemm(x1, x2)

"""
    \.\*(x1::Var, x2::Var)
"""
function .*(x1::Var, x2::Var)
    (x1.data == nothing || x2.data == nothing) && return Var(nothing, .*, (x1,x2))
    length(x1) == length(x2) || throw(DimensionMismatch())
    y = x1.data .* x2.data
    function df(gy)
        isconst(x1) || ∇elemtimes!(gy, x2.data, x1.grad)
        isconst(x2) || ∇elemtimes!(gy, x1.data, x2.grad)
    end
    Var(y, .*, (x1,x2), df)
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
