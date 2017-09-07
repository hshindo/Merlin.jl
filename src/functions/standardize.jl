export Standardize
export standardize

struct Standardize
    scale::Var
    bias::Var
    runmean
    runvar
end

function Standardize{T}(::Type{T}, insize::Tuple)
    dim = 2
    dims = ntuple(i -> i == dim ? 1 : insize[i], length(insize))
    scale = zerograd(ones(T,dims))
    bias = zerograd(zeros(T,dims))
    runmean = zeros(T, dims)
    runvar = ones(T, dims)
    Standardize(scale, bias, runmean, runvar)
end
(f::Standardize)(x) = standardize(x, f.scale, f.bias, f.runmean, f.runvar)

function standardize(x::Var, scale::Var, bias::Var, runmean, runvar; eps=1e-4, decay=0.99)
    T = eltype(x.data)
    xmean = mean(x.data, 2)
    xvar = varm(x.data, xmean, 2)
    n = length(xmean)
    BLAS.scal!(n, T(decay), runmean, 1)
    BLAS.axpy!(T(1-decay), xmean, runmean)
    BLAS.scal!(n, T(decay), runvar, 1)
    a = (1 - decay) * n / max(n-1,1)
    BLAS.axpy!(T(a), xvar, runvar)
    #@. runmean = T(decay) * runmean + T(1-decay) * xmean
    #@. runvar = T(decay) * runvar + T(1-decay) * xvar
    invstd = T(1) ./ sqrt.(xvar + T(eps))
    xhat = (x.data .- xmean) .* invstd
    data = xhat .* scale.data .+ bias.data
    Var(data, standardize, (x,scale,bias,invstd,xhat))
end
standardize(x::Node, scale, bias, runmean, runvar) = Node(standardize, x, scale, bias, runmean, runvar)

function standardize{T}(x::Array{T}, scale::Var, bias::Var, runmean, runvar; eps=1e-4)
    (x .- runmean) ./ sqrt.(runvar + T(eps)) .* scale.data .+ bias.data
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
