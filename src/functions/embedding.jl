export Embedding

type Embedding <: Functor
    w::Var
    idset::IntSet
end

Embedding(w::Var) = Embedding(w, IntSet())
Embedding(w::UniArray) = Embedding(Param(w))

"""
    Embedding(path::String)

Construct Embedding from HDF5 format file.
"""
Embedding(path::String) = Embedding(h5read(path, "Embedding"))

"""
    Embedding{T}(::Type{T}, indim, outdim)

### 👉 Example
```julia
f = Embedding(Float32,10000,100) # 100-length vector, 10k vocabulary
x = Var(rand(1:1000,5,3))
y = f(x)
```
"""
Embedding(T::Type, indim::Int, outdim::Int) = Embedding(rand(T,outdim,indim))

function (f::Embedding)(x::Var)
    y = embedding(f.w.data, x.data)
    function df(gy)
        ∇embedding!(f.w.data, x.data, gy)
        for id in x.data
            push!(f.idset, id)
        end
    end
    Var(y, [x], f, df)
end

function embedding{T}(w::Array{T}, x::Array{Int})
    n = size(w, 1)
    dims = [size(x)...]
    dims[1] *= n
    y = Array(T, dims...)
    for i = 1:length(x)
        copy!(y, (i-1)*n+1, w, (x[i]-1)*n+1, n)
    end
    y
end

function ∇embedding!{T}(gw::Array{T}, x::Array{Int}, gy::Array{T})
    n = size(gw, 1)
    for i = 1:length(x)
        BLAS.axpy!(n, T(1), pointer(gy,(i-1)*n+1), 1, pointer(gw, (x[i]-1)*n+1), 1)
    end
end

function update!(f::Embedding, opt)
    w, idset = f.w, f.idset
    n = size(w.data, 1)
    for id in idset
        o = (id-1) * n + 1
        opt(w.data[o:o+n-1], w.grad[o:o+n-1]) # TODO: use view
    end
    empty!(idset)
end

to_hdf5(f::Embedding) = f.w.data

export quantize!
function quantize!(f::Embedding)
    for w in f.ws
        x = w.data
        for i = 1:length(x)
            x[i] < -0.0 && (x[i] = 0.0)
            x[1] > 1.0 && (x[i] = 1.0)
            x[i] = round(x[i], 1)
        end
    end
end
