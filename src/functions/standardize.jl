export Standardize
export standardize

struct Standardize
    scale::Var
    bias::Var
    eps::Float64
end

function Standardize{T}(::Type{T}, insize::Tuple; eps=1e-4)
    dim = 2
    dims = ntuple(i -> i == dim ? 1 : insize[i], length(insize))
    scale = zerograd(ones(T,dims))
    bias = zerograd(zeros(T,dims))
    Standardize(scale, bias, eps)
end

(f::Standardize)(x::Var) = standardize(x, f.scale, f.bias, f.eps)
(f::Standardize)(x::Node) = Node(standardize, x, f.scale, f.bias)

function standardize(x::Var, scale::Var, bias::Var, eps::Float64)
    if config.train
        T = eltype(x.data)
        m = mean(x.data, 2)
        v = varm(x.data, m, 2)
        invstd = T(1) ./ sqrt.(v + T(eps))
        xhat = (x.data .- m) .* invstd
        data = xhat .* scale.data .+ bias.data
        Var(data, x.batchdims, standardize, (x,scale,bias), work=(invstd,xhat))
    else
        throw("")
    end
end

function addgrad!(y::Var, ::typeof(standardize), x::Var, scale::Var, bias::Var)
    T = eltype(y.data)
    invstd, xhat = y.work
    gscale = sum(y.grad .* xhat, 2)
    gbias = sum(y.grad, 2)

    if !isvoid(x.grad)
        m = size(x.data, 2)
        g = scale.data .* invstd .* (y.grad .- (xhat .* gscale .+ gbias) ./ m)
        BLAS.axpy!(T(1), g, x.grad)
    end
    isvoid(scale.grad) || BLAS.axpy!(T(1), gscale, scale.grad)
    isvoid(bias.grad) || BLAS.axpy!(T(1), gbias, bias.grad)
end
