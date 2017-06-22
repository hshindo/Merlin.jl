export minimize!, makebatch
#import ProgressMeter: Progress, next!

"""
    minimize!(var::Var, opt)

```julia
x = Var()
loss = crossentropy(y, z)

opt = SGD(0.001)
minimize!(loss, opt)
```
"""
function minimize!(var::Var, opt)
    dict = ObjectIdDict()
    nodes = gradient!(var)
    for v in nodes
        isparam(v) && opt(v.data, v.grad)
        f = v[1]
        isa(f,Functor) && (dict[f] = f)
    end
    foreach(f -> update!(f,opt), keys(dict))
end

function makebatch(batchsize::Int, data::Vector...)
    idxs = randperm(length(data[1]))
    r = Var[], Var[], Var[]
    for i in idxs
        batches = map(x -> x[i], data)
        push!(r[1], Var(batches[1],1))
        push!(r[2], Var(batches[2]))
        push!(r[3], Var(batches[3]))
    end
    return r

    for i = 1:batchsize:length(idxs)
        batchidxs = map(k -> idxs[k], i:min(i+batchsize-1,length(idxs)))
        batches = map(x -> x[batchidxs...], data)
    end
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
