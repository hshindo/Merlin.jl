function memalloc(bytesize::Int)
    ref = Ref{Cptr}()
    @apicall :cuMemAlloc (Ptr{Ptr{Cvoid}},Csize_t) ref bytesize
    ref[]
end

function memfree(ptr::Ptr)
    @apicall :cuMemFree (Ptr{Cvoid},) ptr
end

function meminfo()
    ref_free = Ref{Csize_t}()
    ref_total = Ref{Csize_t}()
    @apicall :cuMemGetInfo (Ptr{Csize_t},Ptr{Csize_t}) ref_free ref_total
    Int(ref_free[]), Int(ref_total[])
end
