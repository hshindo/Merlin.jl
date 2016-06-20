sources = [
  "im2col.cpp",
  "maxpooling2d.cpp",
  "softmax.cpp",
  "window2d.cpp",
  "window.cpp"]

compiler = "g++"

@compat if is_windows()
  flags    = ["-fopenmp", "-Wall", "-O3", "-shared", "-march=native"]
  libname = "libmerlin.dll"
  cmd = `$compiler $flags -o $libname $sources`
  println("Running $cmd")
  run(cmd)
elseif is_apple()
  flags    = ["-fPIC", "-Wall", "-O3", "-shared", "-march=native"]
  libname = "libmerlin.so"
  cmd = `$compiler $flags -o $libname $sources`
  println("Running $cmd")
  run(cmd)
else
  flags    = ["-fopenmp", "-fPIC", "-Wall", "-O3", "-shared", "-march=native"]
  libname = "libmerlin.so"
  cmd = `$compiler $flags -o $libname $sources`
  println("Running $cmd")
  run(cmd)
end
