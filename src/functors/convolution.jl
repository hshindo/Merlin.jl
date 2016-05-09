export Convolution

const WINDOW2D_FWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f32)
const WINDOW2D_FWD_F64_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f64)
const WINDOW2D_BWD_F64_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f64)

"""
## Convolution

- `Convolution(l::Linear, strides::NTuple{N,Int}, pads::NTuple{N,Int})

### 👉 Example
```julia
x = Variable(rand(Float32,10,5))
f = Convolution(Linear(Float32,10,7),(10,1),(1,1))
y = f(x)
```
"""
type Convolution{N} <: Functor
  l::Linear
  strides::NTuple{N,Int}
  pads::NTuple{N,Int}
end

@compat (f::Convolution)(arg) = forward(f, arg)

function forward(f::Convolution{2}, arg::Var)
  params = Cint[w1, w2, s1, s2, p1, p2]
end

function backward!(f::Convolution)
end
