export Pooling

#const MAXPOOLING2D_FWD_F32 = Libdl.dlsym(Native.library, :maxpooling2d_fwd_f32)
#const MAXPOOLING2D_BWD_F32 = Libdl.dlsym(Native.library, :maxpooling2d_bwd_f32)

type Pooling{N}
  mode::Symbol
  window::NTuple{N,Int}
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
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
