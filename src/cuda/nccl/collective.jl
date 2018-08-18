function allreduce(sendbuff, recvbuff, count::Int, datatype::Int, op::Int, comm, stream::CuStream)
    @nccl(:ncclAllReduce, (Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint,Cint,Ptr{Cvoid},Ptr{Cvoid}),
        sendbuffm, recvbuff, count, datatype, op, comm, stream)
end

function bcast(buff, count::Int, datatype::Int, root::Int, comm::Comm, stream::CuStream)
    @nccl(:ncclBcast, (Ptr{Cvoid},Csize_t,Cint,Cint,Ptr{Cvoid},Ptr{Cvoid}),
        buff, count, datatype, root, comm, stream)
end

function Base.reduce(sendbuff, recvbuff, count::Int, datatype::Int, op::Int, root::Int, comm::Comm, stream::CuStream)
    @nccl(:ncclReduce, (Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint,Cint,Cint,Ptr{Cvoid},Ptr{Cvoid}),
        sendbuff, recvbuff, count, datatype, op, root, comm, stream)
end
