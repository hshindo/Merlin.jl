const handles = Dict{Int,Ptr{Void}}()

for i = 0:1
    p = Ptr{Void}[0]
    cudnnCreate(p)
    handles[i] = p[1]
end

atexit(() -> begin
    for h in handles
        cudnnDestroy(h)
    end
end)
