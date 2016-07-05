export Data

type Data <: Layer
  y
  gy
end

backward!(l::Data) = ()
tails(l::Data) = []

param(y) = Data(y, zeros(y))

update!(l::Data, opt) = update!(opt, l.y, l.gy)
