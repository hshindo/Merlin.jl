export Linear

"""
    Linear(w::Var, x::Var, [b::Var])

Compute linear function (a.k.a. affine transformation).

```math
f(x) = W^{T}x + b
```
where ``W`` is a weight matrix and ``b`` is a bias vector.

### 👉 Example
```julia
x = Var(rand(Float32,10,5))
f = Linear(Float32,10,7)
y = f(x)
```
"""
type Linear <: Functor
    w::Var
    b::Var
end

function Linear(T::Type, indim::Int, outdim::Int)
    r = T(sqrt(6 / (indim+outdim)))
    w = rand(-r, r, outdim, indim)
    b = fill(T(0), outdim, 1)
    Linear(Var(w), Var(b))
end

function (f::Linear)(x::Var)
    w, b = f.w, f.b
    y = w.data * x.data
    broadcast!(.+, y, y, b.data)
    function df{T}(gy::UniArray{T})
        hasgrad(w) && BLAS.gemm!('N', 'T', T(1), gy, x.data, T(1), w.grad)
        hasgrad(x) && BLAS.gemm!('T', 'N', T(1), w.data, gy, T(1), x.grad)
        for offset = 1:length(b.data):length(gy)
            BLAS.axpy!(length(b.data), T(1), pointer(gy,offset), 1, pointer(b.grad), 1)
        end
    end
    Var(y, [x], f, df)
end

function update!(f::Linear, opt)
    opt(f.w.data, f.w.grad)
    opt(f.b.data, f.b.grad)
end

h5convert(f::Linear) = h5dict(Linear, "w"=>f.w.data, "b"=>f.b.data)

function h5load!(::Type{Linear}, data)
    w, b = Var(data["w"]), Var(data["b"])
    Linear(w, b)
end
