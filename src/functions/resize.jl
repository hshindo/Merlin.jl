export resize

doc"""
    resize(x::Var, batchdims::Vector{Int})

# Example
```julia
T = Float32
x = Var(rand(T,10,5))
y = resize(x, [2,3])
```
"""
function resize(x::Var, batchdims::Vector{Int})
    sum(batchdims) == sum(x.batchdims) || throw("$(batchdims) $(x.batchdims)")
    Var(x.data, batchdims, resize, (x,))
end

resize(x::Node, batchdims; name="") = Node(resize, (x,batchdims), name)

function addgrad!(y::Var, ::typeof(resize), x::Var)
    isvoid(x.grad) || broadcast!(+,x.grad,x.grad,y.grad)
end
