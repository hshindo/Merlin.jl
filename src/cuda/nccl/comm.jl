mutable struct Comm
    ptr::Ptr{Void}
end

Base.unsafe_convert(::Type{Ptr{Void}}, comm::Comm) = comm.ptr

function Comm(nranks::Int, commid, rank::Int)
    ref = Ref{Ptr{Void}}()
    @nccl :ncclCommInitRank (Ptr{Ptr{Void}},Cint,Ptr{UInt8},Cint) ref nranks commid rank
    Comm(ref[])
end

function Comm(ndev::Int, devlist::Vector{Int})
    ref = Ref{Ptr{Void}}()
    @nccl :ncclCommInitAll (Ptr{Ptr{Void}},Cint,Ptr{Cint}) ref ndev devlist
    Comm(ref[])
end

function destroy(comm::Comm)
    @nccl :ncclCommDestroy (Ptr{Void},) comm
end

function get_unique_id()
    ref = Array{UInt8}(NCCL_UNIQUE_ID_BYTES)
    @nccl :ncclGetUniqueId (Ptr{UInt8},) ref
    unsafe_string(ref[])
end

function cudevice(comm::Comm)
    ref = Ref{Cint}()
    @nccl :ncclCommCuDevice (Ptr{Void},Ptr{Cint}) comm ref
    Int(ref[])
end

function userrank(comm::Comm)
    ref = Ref{Cint}()
    @nccl :ncclCommUserRank (Ptr{Void},Ptr{Cint}) comm ref
    Int(ref[])
end
