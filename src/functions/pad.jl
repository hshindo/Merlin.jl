export pad

function pad(x::Array{T}, shapes::Vector{NTuple{N,Int}}; padding=0) where {T,N}
    maxdims = zeros(Int, N)
    for s in shapes
        for i = 1:N
            maxdims[i] < s[i] && (maxdims[i] = s[i])
        end
    end

    y = fill(T(padding), maxdims..., length(shapes))
    st = stride(y, N+1)
    xi = 1
    yi = 1
    for s in shapes
        n = prod(s)
        copy!(y, yi, x, xi, n)
        xi += n
        yi += st
    end
    y
end
