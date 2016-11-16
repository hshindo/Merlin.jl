export Linear

type Linear <: Functor
    w::Var
    b::Var
end

function Linear(T::Type, indim::Int, outdim::Int)
    r = T(sqrt(6 / (indim+outdim)))
    w = rand(T, outdim, indim)
    w .*= 2r
    w .-= r
    b = fill(T(0), outdim, 1)
    Linear(Var(w), Var(b))
end

function (f::Linear)(x::Var)
    w, b = f.w, f.b
    T = eltype(x)
    y = Var(T, (size(w,1),size(x,2)), (x,))
    BLAS.gemm!('N', 'N', T(1), w.data, x.data, T(0), y.data)
    broadcast!(.+, y.data, y.data, b.data)
    y
end
#(f::Linear)(x::GraphNode) = GraphNode(f, x)

function linear(w::Var, x::Var, b::Var)
    y = w.value * x.value
    broadcast!(.+, y, y, b.value)
    function df{T}(gy::Array{T})
        BLAS.gemm!('N', 'T', T(1), gy, x.value, T(1), w.grad)
        BLAS.gemm!('T', 'N', T(1), w.value, gy, T(1), x.grad)
    end
    Var(y, [w,x,b], df)
end

function update!(f::Linear, opt)
    opt(f.w.data, f.w.grad)
    opt(f.b.data, f.b.grad)
end
