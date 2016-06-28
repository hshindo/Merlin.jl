using Compat

const sources = [
  "im2col.cpp",
  "maxpooling2d.cpp",
  "softmax.cpp",
  "window.cpp"]

const compiler = "g++"

@compat if is_windows()
  if haskey(ENV, "MERLIN_BUILD_WINDOWS")
    flags    = ["-fopenmp", "-Wall", "-O3", "-shared", "-march=native"]
    libname = "libmerlin.dll"
    cmd = `$compiler $flags -o $libname $sources`
    println("Running $cmd")
    run(cmd)
  end
elseif is_apple()
  flags    = ["-fPIC", "-Wall", "-O3", "-shared", "-march=native"]
  libname = "libmerlin.so"
  cmd = `$compiler $flags -o $libname $sources`
  println("Running $cmd")
  run(cmd)
elseif is_linux()
  flags    = ["-fopenmp", "-fPIC", "-Wall", "-O3", "-shared", "-march=native"]
  libname = "libmerlin.so"
  cmd = `$compiler $flags -o $libname $sources`
  println("Running $cmd")
  run(cmd)
else
  throw("Unknown OS.")
end
