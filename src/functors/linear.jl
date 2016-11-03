export Linear

type Linear <: Functor
    w::Var
    b::Var
end

function Linear(T::Type, indim::Int, outdim::Int)
    r = T(sqrt(6 / (indim+outdim)))
    w = rand(-r, r, outdim, indim)
    b = fill(T(0), outdim, 1)
    Linear(Var(w), Var(b))
end

(f::Linear)(x::Var) = forward(f, x)

function forward!(f::Linear, v::Var)::Void
    x, y = v[1].data, v.data
    resize!(v, size(f.w,1), size(x,2))
    T = eltype(x)
    BLAS.gemm!('N', 'N', T(1), f.w.data, x, T(1), y)
    #broadcast!(.+, y, y, f.b.data)
end

function backward!(f::Linear, v::Var)
    T = eltype(v.data)
    BLAS.gemm!('N', 'T', T(1), v.grad, v[1].data, T(1), f.w.grad)
    BLAS.gemm!('T', 'N', T(1), f.w.data, v.grad, T(1), v[1].grad)
    # bias
end

function update!(f::Linear, opt)
    opt(f.w.data, f.w.grad)
    opt(f.b.data, f.b.grad)
end
