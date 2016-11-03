export relu

type ReLU <: Functor
end

relu(x::ArrayVar) = ArrayVar(similar(x.data), ReLU(), x)

function forward!(f::ReLU, y::ArrayVar)
    resize!(y, size(y[1]))
    relu!(y.data, y[1].data)
end

backward!(f::ReLU, v::Var) = ∇relu!(v.grad, v[1].data, v[1].grad)

function relu!{T}(y::Array{T}, x::Array{T})
    @inbounds @simd for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end
end

function ∇relu!{T}(gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(x[i] > T(0), gy[i], T(0))
    end
end
