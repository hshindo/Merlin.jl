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

"""
JIT compiler.
- `src`: source code
- `sym`: function name
"""
function compile(src, sym, compiler="g++")
  dir = joinpath(dirname(@__FILE__), "..", "lib")
  symstr = string(sym)
  srcpath = joinpath(dir, "$(symstr).c")
  libname = @windows? "$(symstr).dll" : "$(symstr).so"
  libpath = joinpath(dir, libname)
  if isdefined(sym)
    println("ok")
    Libdl.dlclose(eval(sym))
  end
  open(srcpath, "w") do f
    write(f, src)
  end

  @windows? begin
    run(`$compiler -Wall -O3 -shared -o $libpath $srcpath`)
  end : begin
    run(`$compiler -fPIC -Wall -O3 -shared -o $libpath $srcpath`)
  end

  lib = Libdl.dlopen(libpath)
  h = Libdl.dlsym(lib, :run)
  eval(:(global $sym = $h))
end

end
