export DataIterator
export fit!, evaluate
using Random
using ProgressMeter

mutable struct DataIterator
    dataset
    batchsize
    shuffle
end

function DataIterator(dataset; batchsize::Int, shuffle::Bool)
    DataIterator(dataset, batchsize, shuffle)
end

Base.length(iter::DataIterator) = length(iter.dataset)

function Base.iterate(iter::DataIterator)
	perm = shuffle ? randperm(length(dataset)) : collect(1:length(dataset))
    res = []
    for i = 1:batchsize:length(dataset)
        j = min(i+batchsize-1, length(dataset))
        data = dataset[perm[i:j]]
        push!(res, f(data))
        update!(prog, j)
    end
    res
end
function Base.iterate(iter::DataIterator, state)
	return i >= iter.length ? nothing : (el, (el + it.n, i + 1))
end

function eachbatch(f, dataset, batchsize::Int, shuffle::Bool)
    perm = shuffle ? randperm(length(dataset)) : collect(1:length(dataset))
    prog = Progress(length(dataset))
    res = []
    for i = 1:batchsize:length(dataset)
        j = min(i+batchsize-1, length(dataset))
        data = dataset[perm[i:j]]
        push!(res, f(data))
        update!(prog, j)
    end
    res
end

function fit!(lossfun, model, dataset, opt; batchsize, shuffle)
    settrain(true)
    params = collect(Iterators.flatten(parameters.(graphs(model))))
    res = eachbatch(dataset, batchsize, shuffle) do data
        y = lossfun((model,data))
		loss = sum(Array(y.data))
        gradient!(y)
        foreach(opt, params)
		CUDA.synchronize()
        loss
    end
    loss = sum(res) / length(dataset)
    loss
end

function evaluate(f, model, dataset; batchsize::Int)
    settrain(false)
    eachbatch(dataset, batchsize, false) do data
        y = f((model,data))
		CUDA.synchronize()
		y
    end
end
