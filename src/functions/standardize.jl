export Standardize
export standardize

mutable struct Standardize <: Functor
    scale::Var
    bias::Var
    runmean
    runvar
end

function Standardize(::Type{T}, insize::Tuple) where T
    dim = 2
    dims = ntuple(i -> i == dim ? 1 : insize[i], length(insize))
    scale = zerograd(ones(T,dims))
    bias = zerograd(zeros(T,dims))
    runmean = zeros(T, dims)
    runvar = ones(T, dims)
    Standardize(scale, bias, runmean, runvar)
end
(f::Standardize)(x, train) = standardize(x, train, f.scale, f.bias, f.runmean, f.runvar)

function standardize(x::Var, train::Bool, scale::Var, bias::Var, runmean, runvar; eps=1e-4, decay=0.9)
    T = eltype(x.data)
    if train
        xmean = mean(x.data, 2)
        xvar = varm(x.data, xmean, 2, corrected = size(x.data,2) > 1)
        n = length(xmean)
        @. runmean = T(decay) * runmean + T(1-decay) * xmean
        @. runvar = T(decay) * runvar + T(1-decay) * xvar
        invstd = T(1) ./ sqrt.(xvar + T(eps))
        xhat = (x.data .- xmean) .* invstd
        data = xhat .* scale.data .+ bias.data
        Var(data, (standardize,x,scale,bias,invstd,xhat))
    else
        data = (x.data .- runmean) ./ sqrt.(runvar + T(eps)) .* scale.data .+ bias.data
        Var(data, (standardize,x,scale,bias))
    end
end

function addgrad!(y::Var, ::typeof(standardize), x::Var, scale::Var, bias::Var, invstd, xhat)
    T = eltype(y.data)
    gscale = sum(y.grad .* xhat, 2)
    gbias = sum(y.grad, 2)

    if !isvoid(x.grad)
        n = size(x.data, 2)
        g = scale.data .* (y.grad .- (xhat .* gscale .+ gbias) / n) .* invstd
        BLAS.axpy!(T(1), g, x.grad)
    end
    isvoid(scale.grad) || BLAS.axpy!(T(1), gscale, scale.grad)
    isvoid(bias.grad) || BLAS.axpy!(T(1), gbias, bias.grad)
end
