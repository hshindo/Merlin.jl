export minimize!
import ProgressMeter: Progress, next!

"""
    minimize(output::Var, opt, [progress=true])

```julia
x = Var()
loss = crossentropy(y, z)

opt = SGD(0.001)
minimize!(loss, opt)
```
"""
function minimize!(f, opt, data_x::Vector, data_y::Vector; progress=true, batchsize::Int=10)
    #kwdict = Dict(kwargs)
    length(data_x) == length(data_y) || throw("Length unmatch.")
    progress && (prog = Progress(length(data_x)÷batchsize+1))
    idxs = randperm(length(data_x))
    loss = 0.0
    for i = 1:batchsize:length(idxs)
        batchidxs = i:min(i+batchsize,length(idxs))-1
        y = cat(data_y[r])
        x = cat(data_x[r])
        out = f(x, y)

        loss += out.data
        progress && next!(prog)
    end
    loss / length(data_x)
end

"""
    fit(data_x, data_y, model, opt, [progress=true])
"""
function fit(data_x::Vector, data_y::Vector, model, opt; progress=true)
    length(data_x) == length(data_y) || throw("Length unmatch.")
    progress && (prog = Progress(length(data_x)))
    loss = 0.0
    dict = ObjectIdDict()
    for i in randperm(length(data_x))
        x, y = data_x[i], data_y[i]
        z = model(x, y)
        loss += sum(z.data)
        vars = gradient!(z)
        for v in vars
            isparam(v) && opt(v.data, v.grad)
            isa(v.f, Functor) && (dict[v.f] = v.f)
        end
        foreach(f -> update!(f,opt), keys(dict))
        progress && next!(prog)
    end
    loss /= length(data_x)
    loss
end
