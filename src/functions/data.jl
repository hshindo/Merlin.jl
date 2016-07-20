export Data, Param

type Data <: Var
    data
    grad
    tails::Vector
end

Data(data) = Data(data, nothing, Var[])
Data() = Data(nothing)
Param(data) = Data(data, zeros(data), Var[])

forward(v::Data, x::Var) = Data(x)

backward!(v::Data) = nothing

function update!(v::Data, opt)
  hasgrad(v) && opt(v.data, v.grad)
end
