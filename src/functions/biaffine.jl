export BiAffine
export biaffine

struct BiAffine
    w::Var
    b::Var
end

function BiAffine(::Type{T}, insize::Int, outsize::Int; init_w=Xavier(), init_b=Zeros()) where T
    w = init_w(T, insize, outsize)
    b = init_b(T, outsize)
    Linear(zerograd(w), zerograd(b))
end

(f::Linear)(x) = linear(x, f.w, f.b)

function linear(x::Var, w::Var, b::Var)
    y = linear(x.data, w.data, b.data)
    Var(y, x.batchdims, linear, (x,w,b))
end
linear(x::Node, w::Var, b::Var; name="") = Node(linear, (x,w,b), name)

function linear(x::Matrix, w::Matrix, b::Vector)
    y = gemm('T', 'N', w, x)
    broadcast!(+, y, y, b)
end

function addgrad!(y::Var, ::typeof(linear), x::Var, w::Var, b::Var)
    ∇linear!(y.data, y.grad, x.data, x.grad, w.data, w.grad, b.data, b.grad)
end

function ∇linear!{T}(y::Matrix{T}, gy::Matrix, x::Matrix, gx, w, gw, b, gb)
    isvoid(gx) || BLAS.gemm!('N', 'N', T(1), w, gy, T(1), gx)
    isvoid(gw) || BLAS.gemm!('N', 'T', T(1), x, gy, T(1), gw)
    if !isvoid(gb)
        g = sum(gy, 2)
        BLAS.axpy!(T(1), g, gb)
    end
end
