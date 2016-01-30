export Native
module Native
using ..Merlin

@windows? begin
  const libname = "libmerlin.dll"
end : begin
  const libname = "libmerlin.so"
end

const libpath = abspath(joinpath(dirname(@__FILE__), "..", "deps", libname))
const libpath2 = abspath(joinpath(dirname(@__FILE__), "..", "deps", "helloworld.dll"))

try
  const global library = Libdl.dlopen(libpath)
  const global library2 = Libdl.dlopen(libpath2)
catch y
  println("ERROR: Could not load native extension at $libpath.")
  println("To use native extension, run deps/build.jl to compile the native code.")
  throw(y)
end

end
