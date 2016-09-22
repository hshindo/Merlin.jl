const handles = Ptr{Void}[]

function handle(x::CuArray)
    dev = device(x) + 1
    while dev > length(handles)
        p = Ptr{Void}[0]
        cudnnCreate(p)
        push!(handles, p[1])
    end
    handles[dev]
end

atexit(() -> foreach(cudnnDestroy, handles))
