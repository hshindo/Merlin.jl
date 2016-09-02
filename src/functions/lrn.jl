export lrn

"""
    lrn(x, n, k, alpha, beta)

Local Response Normalization (LRN).

* x::Var: 4-d Var.
* n::Int: Normalization window width.
* k::Float64: Smoothing parameter.
* alpha::Float64: Normalizer scaling parameter.
* beta::Float64: Normalizer power parameter.
"""
function lrn(x::Var, n::Int, k::Float64, alpha::Float64, beta::Float64)
    y = lrn(x.data, n, k, alpha, beta)
    df(gy) = hasgrad(x) && ∇lrn!(x.data, x.grad, y.data, y.grad)
    Var(y, [x], lrn, df)
end

function lrn{T}(x::Array{T,4}, n::Int, k::Float64, alpha::Float64, beta::Float64)
end

lrn(x::CuArray) = JuCUDNN.lrn(x)

function ∇lrn!(x::Array, gx::Array, y::Array, gy::Array)
    throw("Not implemented yet.")
end

∇lrn!(x::CuArray, gx, y, gy) = JuCUDNN.∇lrn!(x, y, gy, gx)
