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

function fit!(lossfun, model, dataset, opt; batchsize, shuffle, device=-1)
    settrain(true)
    params = collect(Iterators.flatten(parameters.(graphs(model))))
    res = eachbatch(dataset, batchsize, shuffle) do data
        y = lossfun((model,data))
		loss = sum(Array(y.data))
        gradient!(y)
        foreach(opt, params)
		device >= 0 && CUDA.synchronize()
        loss
    end
    loss = sum(res) / length(dataset)
    loss
end

function evaluate(f, model, dataset; batchsize, device=-1)
    settrain(false)
    eachbatch(dataset, batchsize, false) do data
        y = f((model,data))
		device >= 0 && CUDA.synchronize()
		y
    end
end
