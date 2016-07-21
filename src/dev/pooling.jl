#const MAXPOOLING2D_FWD_F32 = Libdl.dlsym(Native.library, :maxpooling2d_fwd_f32)
#const MAXPOOLING2D_BWD_F32 = Libdl.dlsym(Native.library, :maxpooling2d_bwd_f32)

"""
    pooling(mode, w, [stride, pad])

## Arguments
* w::Var: weight
* stride::NTuple{N,Int}: stride size. Default: (1,1,...)
* pad::NTuple{N,Int}: padding size. Default: (0,0,...)

## 👉 Example
```julia
x = Var(rand(Float32,5,4,3,2))
f = Conv(Var(rand(Float32,2,2,3,4)), stride=(1,1), pad=(0,0))
y = f(x)
```
"""
function pooling(mode, winsize, stride, padsize, x::Var)
  Pooling(mode,winsize,stride,padsize)(x)
end

type Pooling{N}
  mode::Symbol
  winsize::NTuple{N,Int}
  stride::NTuple{N,Int}
  padsize::NTuple{N,Int}
end

@compat function (f::Pooling)(x::Var)
  @checkargs f (x,)
  if f.mode == :max
    y = maxpooling(f, x.value)
  elseif f.mode == :ave
    y = avepooling(f, x.value)
  end
  df(gy) = hasgrad(x) && ∇pooling!()
  Var(y, df, [x])
end

function maxpooling{T}(f::Pooling, x::Array{T})

end

function avepooling{T}(f::Pooling, x::Array{T})

end
