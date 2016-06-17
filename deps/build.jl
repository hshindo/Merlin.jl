sources = [
  "im2col.cpp",
  "maxpooling2d.cpp",
  "window2d.cpp",
  "window.cpp",
  "math/test_fun.cpp"]

compiler = "g++"

@windows? begin
  flags    = ["-Wall", "-O3", "-shared", "-std=c++11", "-march=native"]
  libname = "libmerlin.dll"
  cmd = `$compiler $flags -o $libname $sources`
  println("Running $cmd")
  run(cmd)
end : begin
  flags    = ["-fPIC", "-Wall", "-O3", "-shared", "-std=c++11", "-march=native"]
  libname = "libmerlin.so"
  cmd = `$compiler $flags -o $libname $sources`
  println("Running $cmd")
  run(cmd)
end
