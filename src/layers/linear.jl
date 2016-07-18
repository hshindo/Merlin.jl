export Linear, linear

@Var(Linear)

function Linear(T::Type, indim::Int, outdim::Int)
    r = T(sqrt(6 / (indim+outdim)))
    w = rand(-r, r, outdim, indim)
    b = fill(T(0), outdim, 1)
    Linear(nothing, nothing, [Param(w),Data(),Param(b)])
end

@compat (f::Linear)(x::Var) = linear(f[1], x, f[3])
@compat (::Linear)(w::Var, b::Var, x::Var) = linear(w, x, b)

function linear(w::Var, x::Var, b::Var)
    !hasdata(w) || !hasdata(b) || !hasdata(x) && return Linear(nothing, nothing, [w,x,b])
    y = w.data * x.data
    broadcast!(.+, y, y, b.data)
    Linear(y, nothing, [w,x,b])
end

backward!(v::Linear) = ∇linear!(v[1], v[2], v[3], v)

function ∇linear!(w::Var, x::Var, b::Var, y::Var)
    ∇times!(w, x, y)
    hasgrad(b) || return
    T = eltype(y.data)
    for offset = 1:length(b.data):length(y.grad)
        BLAS.axpy!(length(b.data), T(1), pointer(y.grad,offset), 1, pointer(b.grad), 1)
    end
end
