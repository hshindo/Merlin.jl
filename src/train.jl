export minimize!
import ProgressMeter: Progress, next!

"""
    minimize!(opt, vars::Var...)

```julia
opt = SGD(0.001)
minimize!(f, opt, data)
```
"""
function minimize!(f, opt, data::Vector; progress=true)
    progress && (prog = Progress(length(data)))
    loss = 0.0
    dict = ObjectIdDict()
    for i in randperm(length(data))
        y = f(data[i])
        loss += sum(y.data)
        nodes = gradient!(y)
        for v in nodes
            if isempty(v.args) && !isvoid(v.grad)
                opt(v.data, v.grad)
            end
            isa(v.f,Functor) && (dict[v.f] = v.f)
        end
        foreach(f -> update!(f,opt), keys(dict))
        progress && next!(prog)
    end
    loss /= length(data)
    loss
end
