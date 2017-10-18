export resize

function resize(x::Var, batchdims::Vector{Int})
    Var(x.data, batchdims, resize, (x,))
end

resize(x::Node, batchdims; name="") = Node(resize, x, batchdims, name=name)

function addgrad!(y::Var, ::typeof(resize), x::Var)
    isvoid(x.grad) || (x.grad .+= y.grad)
end
