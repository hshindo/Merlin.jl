# cudnnBatchNormMode_t
const CUDNN_BATCHNORM_PER_ACTIVATION = Cint(0)
const CUDNN_BATCHNORM_SPATIAL = Cint(1)
const CUDNN_BATCHNORM_SPATIAL_PERSISTENT = Cint(2)

mutable struct ConvolutionDesc
    ptr::Cptr
    work

    function ConvolutionDesc(::Type{T}, pads, strides, dilations) where T
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateConvolutionDescriptor (Ptr{Cptr},) ref
        desc = new(ref[], nothing)
        finalizer(desc, x -> @cudnn :cudnnDestroyConvolutionDescriptor (Cptr,) x.ptr)

        @assert length(pads) == length(strides) == length(dilations)
        N = length(pads)
        cpads = Cint[pads[i] for i=N:-1:1]
        cstrides = Cint[strides[i] for i=N:-1:1]
        cdilations = Cint[dilations[i] for i=N:-1:1]
        mode = CUDNN_CROSS_CORRELATION
        @cudnn(:cudnnSetConvolutionNdDescriptor,
            (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint),
            desc, N, cpads, cstrides, cdilations, mode, datatype(T))
        desc
    end
end

Base.unsafe_convert(::Type{Cptr}, desc::ConvolutionDesc) = desc.ptr


function batchnorm(mode, x::CuArray{T}, scale, bias, mean, var; epsilon=0.001, training=true)
    h = CUDNN.handle(x)
    y = similar(x)
    xdesc = TensorDesc(x)
    sbmvdesc = TensorDesc(scale)
    if inference
        cudnnBatchNormalizationForwardInference(h, mode, T[1], T[0], xdesc, x,
            xdesc, y, sbmvdesc, scale, bias, estimated_mean, estimated_var, Cdouble(epsilon))
    else # training
        running_mean = zeros(scale)
        running_var = zeros(scale)
        save_mean = similar(scale)
        save_invvar = similar(scale)
        cudnnBatchNormalizationForwardTraining(h, mode, T[1], T[0], xdesc, x, ydesc, y,
            sbmvdesc, scale, bias, Cdouble(averagefactor), running_mean, running_var,
    	    Cdouble(epsilon), save_mean, save_invvar)
        function backward!(gy, gx)
            if !isvoid(gx)
                cudnnBatchNormalizationBackward(h, mode, T[1], T[0], T[1], T[0],
                    xdesc, x, dydesc, dy, dxdesc, dx, dsbdesc, scale, dscale, dbias,
                    Cdouble(epsilon), saved_mean, saved_invvar)
            end
        end
    end
    y, backward!
end

function batchnorm_training(mode, x::CuArray{T}, scale::CuArray{T}, bias::CuArray{T},
	running_mean=nothing, running_var=nothing;
	momentum=0.1, epsilon=2e-5, training=true) where T

    h = gethandle()
    y = similar(x)
    xdesc = TensorDesc(x, 4)
    ydesc = TensorDesc(y, 4)
	bndesc = TensorDesc(scale, 4)

	if training
		saved_mean = similar(scale)
	    saved_invvar = similar(scale)
		@cudnn(:cudnnBatchNormalizationForwardTraining,
	        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Csize_t),
			h, mode, T[1], T[0], xdesc, x, ydesc, y, bndesc,
			scale, bias, momentum, running_mean, running_var,
			epsilon, saved_mean, saved_invvar)
	else

	end
    y, (save_mean,save_invvar)
end

function ∇batchnorm!(mode, x, dy, dx, scale, dscale, dbias, epsilon::Float64, saved_mean, saved_invvar)
    T = eltype(x)
    h = handle(x)
    xdesc = tensor_desc(x)
    dydesc = tensor_desc(dy)
    dxdesc = tensor_desc(dx)
    dsbdesc = tensor_desc(scale)

    cudnnBatchNormalizationBackward(h, mode, T[1], T[0], T[1], T[0],
    xdesc, x, dydesc, dy, dxdesc, dx, dsbdesc, scale, dscale, dbias,
        Cdouble(epsilon), saved_mean, saved_invvar)
end
