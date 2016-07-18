export Data, Param

@Var(Data)

Data(data) = Data(data, nothing, Var[])
Data() = Data(nothing)
Param(data) = Data(data, zeros(data), Var[])

forward(v::Data, x::Var) = Data(x)

backward!(v::Data) = nothing

function update!(opt, v::Data)
  hasgrad(v) && update!(opt, v.data, v.grad)
end
