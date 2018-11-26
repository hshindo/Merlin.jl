export minimize!, evaluate

function minimize!(f, data::DataLoader, opt)
    params = parameters(f)
    loss = 0.0
    foreach(data) do x
        out = f(x)
        data.device >= 0 && CUDA.synchronize()
        loss += sum(Array(out.data))
        gradient!(out)
        data.device >= 0 && CUDA.synchronize()
        opt.(params)
    end
    loss
end

function evaluate(f, data::DataLoader)
    res = []
    foreach(data) do x
        y = f(x)
        push!(res, y)
        data.device >= 0 && CUDA.synchronize()
    end
    res
end
