# Graph
`Graph` is a container of computational graph.

```julia
ls = [Linear(Float32,10,5), Linear(FLoat32,5,3)]
f = @graph (:x,) begin
  x = :x
  x = ls[1](x)
  x = relu(x)
  x = ls[2](x)
  x
end
```

Then, apply the function `g` to input var.
```julia
x = Var(rand(Float32,10,10))
y = f(x)
```
