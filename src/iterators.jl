export BatchIterator

struct BatchIterator
    data::Vector
    batchsize::Int
    backend
    shuffle::Bool
end

function BatchIterator(data::Vector, batchsize; backend=nothing, shuffle=true)
    if backend != nothing
        data = map(data) do x
            map(backend, x)
        end
    end
    BatchIterator(data, batchsize, backend, shuffle)
end

Base.getindex(iter::BatchIterator, i::Int) = iter.data[i]

Base.eltype(iter::BatchIterator) = eltype(iter[1])

function Base.start(iter::BatchIterator)
    iter.shuffle && shuffle!(iter.data)
    1
end

function Base.next(iter::BatchIterator, i::Int)
    iter.batchsize == 1 && return iter[i], i+1

    j = min(i+iter.batchsize-1, length(iter.data))
    data = iter.data[i:j]
    @assert isa(data[1], Tuple)
    batch = ntuple(length(data[1])) do k
        cat(ndims(data[1][k]), map(x -> x[k], data)...)
    end
    batch, j+1
end

Base.done(iter::BatchIterator, i::Int) = i > length(iter.data)
