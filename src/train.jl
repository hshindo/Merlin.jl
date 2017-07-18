export minimize!
import ProgressMeter: Progress, next!

"""
    minimize!(opt, vars::Var...)

```julia
x = Var()
loss = crossentropy(y, z)

opt = SGD(0.001)
minimize!(opt, loss)
```
"""
function minimize!(opt, vars::Var...)
    dict = ObjectIdDict()
    nodes = gradient!(vars...)
    for v in nodes
        isempty(v.args) && !isvoid(v.grad) && opt(v.data,v.grad)
        f = v.f
        isa(f,Functor) && (dict[f] = f)
    end
    foreach(f -> update!(f,opt), keys(dict))
end

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

function minimize2!(; batchsize=10)
    Threads.@threads for i = 1:batchsize
        y[i] = f(x[i]...)
    end
end
