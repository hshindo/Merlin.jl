const sources = [
    "im2col.cpp",
    "pooling.cpp"]

if Sys.iswindows()
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
    cmd = `$gpp -Wall -O3 -fPIC -shared -I $incdir -o libmerlin.dll $sources`
elseif Sys.isapple()
    cmd = `g++ -Wall -O3 -fPIC -shared -o libmerlin.so $sources`
elseif Sys.islinux()
    cmd = `g++ -Wall -O3 -fPIC -shared -o libmerlin.so $sources`
else
    throw("Unknown OS.")
end

println(cmd)
run(cmd)
println()
println("Build success.")
