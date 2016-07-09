immutable CArray{T}
  ptr::Ptr{T}
  dims::Ptr{Cint}
  strides::Ptr{Cint}
end

"""
JIT C++ compiler.
- `src`: source code
- `sym`: function name
"""
function cppcompile(src, sym::Symbol)
  dir = joinpath(dirname(@__FILE__), "..", "lib")
  symstr = string(sym)
  srcpath = joinpath(dir, "$(symstr).c")
  libname = @windows? "$(symstr).dll" : "$(symstr).so"
  libpath = joinpath(dir, libname)
  #Libdl.dlclose(eval(sym))

  compiler = "g++"
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
  @eval global $sym = $h
end
