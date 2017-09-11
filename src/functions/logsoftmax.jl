export logsoftmax

const LOGSOFTMAX_F32 = Libdl.dlsym(libmerlin, :logsoftmax_f32)
const ∇LOGSOFTMAX_F32 = Libdl.dlsym(libmerlin, :logsoftmax_f32_grad)

logsoftmax_handle(::Type{Float32}) = LOGSOFTMAX_F32
∇logsoftmax_handle(::Type{Float32}) = ∇LOGSOFTMAX_F32

"""
    logsoftmax(x)

Logarithm of softmax function.
"""
logsoftmax(x::Var) = Var(logsoftmax(x.data), x.batchdims, logsoftmax, (x,))

logsoftmax(x::Node; name) = Node(logsoftmax, x, name=name)

function logsoftmax{T}(x::Array{T})
    y = similar(x)
    h = logsoftmax_handle(T)
    dims = size3d(x, ndims(x)-1)
    ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint), x, y, dims[1], dims[2], dims[3])
    y
end

function addgrad!(y::Var, ::typeof(logsoftmax), x::Var)
    isvoid(x.grad) || ∇logsoftmax!(y.data, y.grad, x.grad)
end

function ∇logsoftmax!{T}(y::Array{T}, gy::Array{T}, gx::Array{T})
    h = ∇logsoftmax_handle(T)
    dims = size3d(y, ndims(y)-1)
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{T},Cint,Cint,Cint), y, gy, gx, dims[1], dims[2], dims[3])
end

function logsoftmax_jl{T}(x::Matrix{T})
    y = similar(x)
    max = maximum(x, 1)
    for j = 1:size(x,2)
        sum = T(0)
        @inbounds for i = 1:size(x,1)
            sum += exp(x[i,j] - max[j])
        end
        logz = log(sum)
        @inbounds for i = 1:size(x,1)
            y[i,j] = x[i,j] - max[j] - logz
        end
    end
    y
end
