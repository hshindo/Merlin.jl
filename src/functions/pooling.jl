export maxpooling, avepooling

const MAXPOOLING2D_FWD_F32 = Libdl.dlsym(Native.library, :maxpooling2d_fwd_f32)
const MAXPOOLING2D_BWD_F32 = Libdl.dlsym(Native.library, :maxpooling2d_bwd_f32)

type Pooling{N}
  mode # max or average
  window::NTuple{N,Int}
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
end

@compat function (f::Pooling)(x::Var)
  @checkargs f (x,)
  throw("Not implemented yet.")
end
