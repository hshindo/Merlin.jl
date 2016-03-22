# Overview
Merlin.jl provides many primitive functions.

Every computation is preserved.

## Decoding
```julia
v = Variable()
```

## Training
```julia
opt = SGD(0.001)

for i = 1:10
  v = Variable()
  y = v |> f1 |> f2
  y.grad = ones()
  backward!(y)
  update!(opt, y)
end
```
