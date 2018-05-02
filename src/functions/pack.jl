export pack, unpack

function pack(xs::Vector; pad=0)
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
        copy!(y, yi, x, 1)
        yi += st
    end
    y
end

function unpack(x::A, sizes::Vector) where A
    ys = map(A, sizes)
    for i = 1:length(sizes)
        ys[i] = x[s...,i]
    end
    ys
end
Array{Float32,2}(2,3)
