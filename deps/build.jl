sources = ["maxpool2d.cpp", "window2d.cpp", "crossentropy.cpp"]

compiler = "g++"

@windows? begin
  flags    = ["-Wall", "-O3", "-shared", "-fopenmp"]
  libname = "libmerlin.dll"
  cmd = `$compiler $flags -o $libname $sources`
  println("Running $cmd")
  run(cmd)
end : begin
  flags    = ["-fPIC", "-Wall", "-O3", "-shared"]
  libname = "libmerlin.so"
  cmd = `$compiler $flags -o $libname $sources`
  println("Running $cmd")
  run(cmd)
end
