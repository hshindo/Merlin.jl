function uniform{T}(::Type{T}, a, b, dims::Tuple)
    a < b || throw("Invalid interval: [$a: $b]")
    r = rand(T, dims)
    r .*= T(b - a)
    r .+= T(a)
    r
end
uniform{T}(::Type{T}, a, b, dims::Int...) = uniform(T, a, b, dims)

function redim{T,N}(x::Array{T,N}, n::Int; pad=0)
    dims = ntuple(n) do i
        1 <= i-pad <= N ? size(x,i-pad) : 1
    end
    reshape(x, dims)
end

function minibatch(data::Vector, size::Int)
    batches = []
    for i = 1:size:length(data)
        xs = [data[k] for k = i:min(i+size-1,length(data))]
        b = cat(ndims(data[1])+1, xs...)
        push!(batches, b)
    end
    T = typeof(batches[1])
    Vector{T}(batches)
end

add!(y::Array, x::Array) = BLAS.axpy!(a, x, y)
function add!(a, x::SubArray, y)
    throw("Not implemented yet.")
end
