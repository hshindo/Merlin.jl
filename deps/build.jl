const sources = [
    "im2col.cpp",
    "pooling.cpp",
    "softmax.cpp",
    "window.cpp"]

const compiler = "g++"

if is_windows()
    builddir = dirname(Base.source_path())
    println("Build directory is $(builddir)")

    import WinRPM

    println("Installing gcc-c++.")
    WinRPM.install("gcc-c++"; yes=true)
    WinRPM.install("gcc"; yes=true)
    WinRPM.install("headers"; yes=true)

    gpp = Pkg.dir("WinRPM","deps","usr","x86_64-w64-mingw32","sys-root","mingw","bin","g++")
    RPMbindir = Pkg.dir("WinRPM","deps","usr","x86_64-w64-mingw32","sys-root","mingw","bin")
    incdir = Pkg.dir("WinRPM","deps","usr","x86_64-w64-mingw32","sys-root","mingw","include")

    push!(Base.Libdl.DL_LOAD_PATH,RPMbindir)
    ENV["PATH"] = ENV["PATH"] * ";" * RPMbindir

    run(`$gpp --version`)
    cmd = `$gpp -Wall -shared -O3 -I $incdir -o libmerlin.dll $sources`
    println(cmd)
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
