export Data, Param

type Data <: Var
  data
  grad
  tails::Vector{Var}
end

Data(data) = Data(data, nothing, Var[])
Data() = Data(nothing)
Param(data) = Data(data, zeros(data), Var[])

forward(v::Data, x::Var) = Data(x)

backward!(v::Data) = nothing

function update!(opt, v::Data)
  hasgrad(v) && update!(opt, v.data, v.grad)
end
