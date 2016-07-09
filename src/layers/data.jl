export Data

type Data <: Layer
  data
  grad
  name::Symbol
end

Data(data) = Data(data, nothing)
Data(data, grad) = Data(data, grad, gensym())
Data(; name=gensym()) = Data(nothing, nothing, name)

tails(l::Data) = Layer[]

forward!(l::Data) = nothing
backward!(l::Data) = nothing

update!(l::Data, opt) = update!(opt, l.y, l.gy)

function to_cuda(l::Data)
  l.y = CuArray(l.y)
end
