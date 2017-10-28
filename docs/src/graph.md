# Graph
`Graph` represents a computational graph.

```julia
using Merlin

T = Float32
x = Node()
y = Linear(T,10,7)(x)
y = relu(y)
y = Linear(T,7,3)(y)
@assert typeof(y) == Node
g = Graph(input=x, output=y)

x = zerograd(rand(T,10,10))
y = g(x)

params = gradient!(y)
println(x.grad)

opt = SGD(0.01)
foreach(opt, params)
```
