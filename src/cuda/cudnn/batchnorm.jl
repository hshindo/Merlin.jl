# cudnnBatchNormMode_t
const CUDNN_BATCHNORM_PER_ACTIVATION = 0
const CUDNN_BATCHNORM_SPATIAL = 1
const CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2
# cudnnBatchNormOps_t
CUDNN_BATCHNORM_OPS_BN = 0 # do batch normalization only
CUDNN_BATCHNORM_OPS_BN_ACTIVATION = 1 # do batchNorm, then activation
CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION = 2 # do batchNorm, then elemWiseAdd, then activation

struct BatchNorm
	mode
	x
	scale
	epsilon
	save_mean
	save_invvar
end

function batchnorm!(mode, x::CuArray{T,4}, scale::CuArray{T,4}, bias::CuArray{T,4}, running_mean, running_var;
	momentum=0.1, epsilon=2e-5, training::Bool) where T

    h = gethandle()
    y = similar(x)
    xdesc = TensorDesc(x, 4)
    ydesc = TensorDesc(y, 4)
	bndesc = TensorDesc(scale, 4)
	if training
		save_mean = similar(scale)
		save_invvar = similar(scale)
		@cudnn(:cudnnBatchNormalizationForwardTraining,
			(Cptr,Cint,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,
			Cptr,Cptr,Cptr,
			Cdouble,Cptr,Cptr,
			Cdouble,Cptr,Cptr),
			h, mode, T[1], T[0], xdesc, x, ydesc, y,
			bndesc, scale, bias,
			momentum, running_mean, running_var,
			epsilon, save_mean, save_invvar)
		y, BatchNorm(mode,x,scale,epsilon,save_mean,save_invvar)
	else
		@cudnn(:cudnnBatchNormalizationForwardInference,
			(Cptr,Cint,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,
			Cptr,Cptr,Cptr,
			Cptr,Cptr,Cdouble),
			h, mode, T[1], T[0], xdesc, x, ydesc, y,
			bndesc, scale, bias,
			running_mean, running_var, epsilon)
		y, nothing
	end
end

function ∇batchnorm!(bn::BatchNorm, dy::CuArray{T}, dx, dscale, dbias) where T
	mode, x, scale = bn.mode, bn.x, bn.scale
	h = gethandle()
    xdesc = TensorDesc(x, 4)
    dydesc = TensorDesc(dy, 4)
	bndesc = TensorDesc(scale, 4)
	@cudnn(:cudnnBatchNormalizationBackward,
		(Cptr,Cint,Cptr,Cptr,Cptr,Cptr,
		Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,
		Cdouble,Cptr,Cptr),
		h, mode, T[1], T[1], T[1], T[1],
		xdesc, x, dydesc, dy, xdesc, dx, bndesc, scale, dscale, dbias,
		bn.epsilon, bn.save_mean, bn.save_invvar)
end
