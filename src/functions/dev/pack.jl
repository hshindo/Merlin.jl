export pack, unpack

function pack(xs::Vector{Var}; pad=0)
    T = eltype(xs[1])
    N = ndims(xs[1])
    maxdims = zeros(Int, N)
    for x in xs
        s = size(x)
        for i = 1:N
            maxdims[i] < s[i] && (maxdims[i] = s[i])
        end
    end

    y = fill(T(pad), maxdims..., length(xs))
    st = stride(y, N+1)
    yi = 1
    for x in xs
        copyto!(y, yi, x.data, 1)
        yi += st
    end
    Var(y, (pack,xs))
end
pack(x::Node) = Node(pack, x)

function unpack(x::A, sizes::Vector) where A
    ys = map(A, sizes)
    for i = 1:length(sizes)
        ys[i] = x[s...,i]
    end
    ys
end
