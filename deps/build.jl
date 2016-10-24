const sources = [
    "im2col.cpp",
    "pooling.cpp",
    "softmax.cpp",
    "window.cpp"]

const compiler = "g++"

if is_windows()
    flags    = ["-Wall", "-O3", "-shared", "-march=native"]
    libname = "libmerlin.dll"
    cmd = `$compiler $flags -o $libname $sources`
    println("Running $cmd")
    run(cmd)
elseif is_apple()
    flags    = ["-fPIC", "-Wall", "-O3", "-shared"]
    libname = "libmerlin.so"
    cmd = `$compiler $flags -o $libname $sources`
    println("Running $cmd")
    run(cmd)
elseif is_linux()
    flags    = ["-fPIC", "-Wall", "-O3", "-shared", "-march=native"]
    libname = "libmerlin.so"
    cmd = `$compiler $flags -o $libname $sources`
    println("Running $cmd")
    run(cmd)
else
    throw("Unknown OS.")
end
