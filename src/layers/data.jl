export Data, Param

type Data <: Var
    data
    grad
    tails::Vector
end

Data(data) = Data(data, nothing, Var[])
Data() = Data(nothing)
Param(data) = Data(data, zeros(data), Var[])

backward!(v::Data) = nothing

function update!(v::Data, opt)
  v.grad == nothing || opt(v.data, v.grad)
end
