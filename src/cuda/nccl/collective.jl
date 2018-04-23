function allreduce(sendbuff, recvbuff, count::Int, datatype::Int, op::Int, comm, stream::CuStream)
    @nccl(:ncclAllReduce, (Ptr{Void},Ptr{Void},Csize_t,Cint,Cint,Ptr{Void},Ptr{Void}),
        sendbuffm, recvbuff, count, datatype, op, comm, stream)
end

function bcast(buff, count::Int, datatype::Int, root::Int, comm::Comm, stream::CuStream)
    @nccl(:ncclBcast, (Ptr{Void},Csize_t,Cint,Cint,Ptr{Void},Ptr{Void}),
        buff, count, datatype, root, comm, stream)
end

function Base.reduce(sendbuff, recvbuff, count::Int, datatype::Int, op::Int, root::Int, comm::Comm, stream::CuStream)
    @nccl(:ncclReduce, (Ptr{Void},Ptr{Void},Csize_t,Cint,Cint,Cint,Ptr{Void},Ptr{Void}),
        sendbuff, recvbuff, count, datatype, op, root, comm, stream)
end
