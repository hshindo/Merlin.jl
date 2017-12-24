box(x) = x
box(x::Int) = Cint(x)
box{N}(t::NTuple{N,Int}) = map(Cint, t)
box(x::Vector{Int}) = ntuple(i -> Cint(x[i]), length(x))

function (f::CuFunction)(args...;
    dx=1, dy=1, dz=1, bx=128, by=1, bz=1, sharedmem=0, stream=C_NULL)
    argptrs = Ptr{Void}[pointer_from_objref(box(a)) for a in args]
    gx = ceil(dx / bx)
    gy = ceil(dy / by)
    gz = ceil(dz / bz)
    cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sharedmem, stream, argptrs, stream)
end
