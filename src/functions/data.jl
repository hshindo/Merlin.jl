export Data, Param

type Data <: Var
    data
    grad
    tails::Vector
end

Data(data) = Data(data, nothing, Var[])
Data() = Data(nothing)
Data(T::Type, indim::Int, outdim::Int)
Param(data) = Data(data, zeros(data), Var[])

backward!(v::Data) = nothing

function update!(v::Data, opt)
  v.grad == nothing || opt(v.data, v.grad)
end
