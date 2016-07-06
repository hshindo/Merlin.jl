export Data

type Data <: Layer
  y
  gy
end

Data(y) = Data(y, nothing)

tails(l::Data) = []

backward!(l::Data) = nothing

update!(l::Data, opt) = update!(opt, l.y, l.gy)
