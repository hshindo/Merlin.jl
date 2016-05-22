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

function compile(code, name)
  path = joinpath(dirname(@__FILE__), "..", "deps")
  open(joinpath(path,"$(name).c"), "w") do f
    write(f, code)
  end
  run(`gcc -fPIC -Wall -O3 -shared -o $(name).dll $path`)
  #lib = Libdl.dlopen(libpath2)
  #h = Libdl.dlsym(lib, symbol(name))
  #libdict["a"] = h
end

"""
Create a global symbol for ccall
"""
macro dlsym(func, lib)
  z, zlocal = gensym(string(func)), gensym()
  eval(current_module(),:(global $z = C_NULL))
  z = esc(z)
  quote
    let $zlocal::Ptr{Void} = $z::Ptr{Void}
      if $zlocal == C_NULL
        $zlocal = dlsym($(esc(lib))::Ptr{Void}, $(esc(func)))
        global $z = $zlocal
      end
      $zlocal
    end
  end
end

end
