mutable struct Comm
    ptr::Ptr{Cvoid}
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, comm::Comm) = comm.ptr

function Comm(nranks::Int, commid, rank::Int)
    ref = Ref{Ptr{Cvoid}}()
    @nccl :ncclCommInitRank (Ptr{Ptr{Cvoid}},Cint,Ptr{UInt8},Cint) ref nranks commid rank
    Comm(ref[])
end

function Comm(ndev::Int, devlist::Vector{Int})
    ref = Ref{Ptr{Cvoid}}()
    @nccl :ncclCommInitAll (Ptr{Ptr{Cvoid}},Cint,Ptr{Cint}) ref ndev devlist
    Comm(ref[])
end

function destroy(comm::Comm)
    @nccl :ncclCommDestroy (Ptr{Cvoid},) comm
end

function get_unique_id()
    ref = Array{UInt8}(NCCL_UNIQUE_ID_BYTES)
    @nccl :ncclGetUniqueId (Ptr{UInt8},) ref
    unsafe_string(ref[])
end

function cudevice(comm::Comm)
    ref = Ref{Cint}()
    @nccl :ncclCommCuDevice (Ptr{Cvoid},Ptr{Cint}) comm ref
    Int(ref[])
end

function userrank(comm::Comm)
    ref = Ref{Cint}()
    @nccl :ncclCommUserRank (Ptr{Cvoid},Ptr{Cint}) comm ref
    Int(ref[])
end
