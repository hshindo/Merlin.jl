export BatchedData

struct BatchedData
    data::Vector
    batchsize::Int
    backend
    shuffle::Bool
end

function BatchedData(data::Vector, batchsize::Int; backend=CPUIBackend(), shuffle=true)
    data = map(data) do x
        map(backend, x)
    end
    Batch(data, batchsize, backend, shuffle)
end

Base.getindex(iter::Batch, i::Int) = iter.data[i]

Base.eltype(iter::Batch) = eltype(iter[1])

function Base.start(iter::Batch)
    iter.shuffle && shuffle!(iter.data)
    1
end

function Base.next(iter::Batch, i::Int)
    iter.batchsize == 1 && return iter[i], i+1

    j = min(i+iter.batchsize-1, length(iter.data))
    data = iter.data[i:j]
    @assert isa(data[1], Tuple)
    batch = ntuple(length(data[1])) do k
        cat(ndims(data[1][k]), map(x -> x[k], data)...)
    end
    batch, j+1
end

Base.done(iter::Batch, i::Int) = i > length(iter.data)
