# cudnnActivationMode_t
const CUDNN_ACTIVATION_SIGMOID = Cint(0)
const CUDNN_ACTIVATION_RELU = Cint(1)
const CUDNN_ACTIVATION_TANH = Cint(2)
const CUDNN_ACTIVATION_CLIPPED_RELU = Cint(3)
const CUDNN_ACTIVATION_ELU = Cint(4)

"""
    ActivationDesc

* coef: floating point number to specify the clipping threashold when the activation
mode is set to CUDNN_ACTIVATION_CLIPPED_RELU or to specify the alpha coefficient
when the activation mode is set to CUDNN_ACTIVATION_ELU
"""
mutable struct ActivationDesc
    ptr::Cptr

    function ActivationDesc(mode::Cint, coef::Float64)
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateActivationDescriptor (Ptr{Cptr},) ref
        desc = new(ref[])
        finalizer(desc) do x
            @cudnn :cudnnDestroyActivationDescriptor (Cptr,) x.ptr
        end

        @cudnn(:cudnnSetActivationDescriptor,
            (Cptr,Cint,Cint,Cdouble),
            desc, mode, CUDNN_NOT_PROPAGATE_NAN, coef)
        desc
    end
end

const ACTIVATION_DESCS = Dict{Tuple,ActivationDesc}()

Base.unsafe_convert(::Type{Cptr}, desc::ActivationDesc) = desc.ptr

function activation(x::CuArray{T}, mode::Cint, coef::Float64) where T
    actdesc = get!(ACTIVATION_DESCS, (mode,coef)) do
        ActivationDesc(mode, coef)
    end
    xdesc = TensorDesc(x, 4)
    y = similar(x)
    @cudnn(:cudnnActivationForward,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr),
        gethandle(), actdesc, T[1], xdesc, x, T[0], xdesc, y)
    y
end

sigmoid(x::CuArray) = activation(x, CUDNN_ACTIVATION_SIGMOID, 0.0)
relu(x::CuArray) = activation(x, CUDNN_ACTIVATION_RELU, 0.0)
tanh(x::CuArray) = activation(x, CUDNN_ACTIVATION_TANH, 0.0)

function ∇activation!(y::CuArray{T}, dy, x, dx, mode, coef) where T
    actdesc = ActivationDesc(mode, coef)
    xdesc = TensorDesc(x, 4)
    @cudnn(:cudnnActivationBackward,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,
        Cptr,Cptr,Cptr,Cptr,Cptr),
        gethandle(), actdesc, T[1], xdesc, y, xdesc, dy, xdesc, x, T[1], xdesc, dx)
end

∇sigmoid!(y, dy, x, dx) = ∇activation!(y, dy, x, dx, CUDNN_ACTIVATION_SIGMOID, 0.0)
∇relu!(y, dy, x, dx) = ∇activation!(y, dy, x, dx, CUDNN_ACTIVATION_RELU, 0.0)
∇tanh!(y, dy, x, dx) = ∇activation!(y, dy, x, dx, CUDNN_ACTIVATION_TANH, 0.0)
