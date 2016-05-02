export Native
module Native

@windows? begin
  const libname = "libmerlin.dll"
end : begin
  const libname = "libmerlin.so"
end

const libpath = abspath(joinpath(dirname(@__FILE__), "..", "deps", libname))

try
  const global library = Libdl.dlopen(libpath)
catch y
  println("ERROR: Could not load native extension at $libpath. Try `Pkg.build("Merlin.jl")` to compile native codes.")
  throw(y)
end

const SOFTMAX_F32 = Libdl.dlsym(library, :softmax_fw_f32)
#const WINDOW2D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f32)
#const WINDOW2D_FWD_F64_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f64)
#const WINDOW2D_BWD_F64_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f64)

softmax(x::Float32) = ccall(SOFTMAX_F32, Void, (Cfloat,), x)

end
