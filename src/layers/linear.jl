export Linear

type Linear <: Functor
    w::Array
    b::Array
end

function Linear(T::Type, indim::Int, outdim::Int)
    r = T(sqrt(6 / (indim+outdim)))
    w = rand(T, outdim, indim)
    w .*= 2r
    w .-= r
    b = fill(T(0), outdim, 1)
    Linear(w, b)
end

function forward!(f::Linear, l::Layer)
    #l.data = similar(f.w, size(f.w,1), size(l[1].data,2))
    T = eltype(l.data)
    #resize!(y, size(f.w,1), size(y[1],2))
    BLAS.gemm!('N', 'N', T(1), f.w, l[1].data, T(0), l.data)
    #broadcast!(.+, y, y, f.b.data)
end

function backward!(f::Linear, y)
    BLAS.gemm!('N', 'T', T(1), y.grad, y[1].data, T(1), f.w.grad)
    BLAS.gemm!('T', 'N', T(1), f.w.data, y.grad, T(1), y[1].grad)
    # bias
end

function update!(f::Linear, opt)
    opt(f.w.data, f.w.grad)
    opt(f.b.data, f.b.grad)
end
