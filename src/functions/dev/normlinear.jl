
export NormLinear
struct NormLinear
    w
    b
end

function NormLinear{T}(::Type{T}, insize::Int, outsize::Int)
    w = randn(T,outsize,insize) * T(1 / insize)
    b = fill(T(0), outsize)
    Linear(zerograd(w), zerograd(b))
end

(f::NormLinear)(x) = normlinear(f.w, x, f.b)

function normlinear(w::Var, x::Var, b::Var)
    y = w.data * x.data .+ b.data
    Var(y, normlinear, (w,x,b))
end

function addgrad!(y::Var, ::typeof(normlinear), w::Var, x::Var, b::Var)
    T = eltype(y.data)
    isvoid(w.grad) || BLAS.gemm!('N', 'T', T(1), y.grad, x.data, T(1), w.grad)
    isvoid(x.grad) || BLAS.gemm!('T', 'N', T(1), w.data, y.grad, T(1), x.grad)
    if !isvoid(b.grad)
        g = sum(y.grad, 2)
        BLAS.axpy!(T(1), g, b.grad)
    end
end

#=
export NormLinear
type NormLinear
    v::Var
    g::Var
    b::Var
end

function NormLinear{T}(::Type{T}, indim::Int, outdim::Int)
    v = zerograd(randn(T,indim,outdim) * 0.05)
    g = zerograd(ones(T,1,outdim))
    b = zerograd(zeros(T,outdim))
    NormLinear(v, g, b)
end

(f::NormLinear)(x::Var) = normlinear(x, f.v, f.g, f.b)

function normlinear(x::Var, v::Var, g::Var, b::Var)
    w = normalize(v) .* g
    linear(x, w, b)
end
=#
