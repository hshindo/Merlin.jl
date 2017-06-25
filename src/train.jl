export minimize!

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
        isa(f,Functor) && (dict[f] = f)
    end
    foreach(f -> update!(f,opt), keys(dict))
end

function minimize!(f, data::Vector, opt)
    dict = ObjectIdDict()
    nodes = gradient!(vars...)
    for v in nodes
        isempty(v.args) && !isvoid(v.grad) && opt(v.data,v.grad)
        isa(f,Functor) && (dict[f] = f)
    end
    foreach(f -> update!(f,opt), keys(dict))
end

function minimize2!(; batchsize=10)

    Threads.@threads for i = 1:batchsize
        y[i] = f(x[i]...)
    end
end
