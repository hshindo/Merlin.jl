export Data

type Data <: Layer
  y
  gy
end

Data(y) = Data(y, nothing)

backward!(l::Data) = nothing
tails(l::Data) = []

update!(l::Data, opt) = update!(opt, l.y, l.gy)
