export pad

function pad(xs::Vector{Var}, padding)
    T = eltype(xs[1])
    N = ndims(xs[1])
    maxdims = zeros(Int, N)
    for x in xs
        for d = 1:N
            maxdims[d] < size(x,d) && (maxdims[d] = size(x,d))
        end
    end

    y = similar(xs[1].data, maxdims..., length(xs))
    fill!(y, T(padding))
    st = stride(y, N+1)
    yi = 1
    for x in xs
        copy!(y, yi, x.data, 1, length(x))
        yi += st
    end
    Var(y, (pad,xs,padding))
end
pad(xs::Node, padding) = Node(pad, xs, padding)

function addgrad!(y::Var, ::typeof(pad), xs::Vector{Var}, padding)
    st = stride(y, ndims(y))
    yi = 1
    for x in xs
        isvoid(x.grad) && continue
        add!(x.grad, 1, y.grad, yi, length(x))
        yi += st
    end
end
