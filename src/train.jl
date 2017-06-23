export minimize!

"""
    minimize!(var::Var, opt)

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
        if !isempty(v.args)
            f = v[1]
            isa(f,Functor) && (dict[f] = f)
        end
    end
    foreach(f -> update!(f,opt), keys(dict))
end
