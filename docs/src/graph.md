# Graph
`Graph` is a container of computational graph.

```julia
f = @graph begin
  x = Var(:x)
  x = LinearFun(Float32,10,5)
  x = relu(x)
  x = LinearFun(FLoat32,5,3)
  x
end
```

Then, apply the function `g` to input var.
```julia
x = Var(rand(Float32,10,10))
y = f(:x => x)
```
