# Initializers

```@index
Pages = ["initializers.md"]
```

```@docs
Uniform
Normal
Xavier
Orthogonal
```

## Custom Initializer
```julia
import Merlin.random

struct CustomRand
end

function random{T}(init, ::Type{T}, dims...)
    # code
end
```
