export
    CUDNN_BATCHNORM_PER_ACTIVATION,
    CUDNN_BATCHNORM_SPATIAL

function batchnorm_inference(mode, x, scale, bias, estimated_mean, estimated_var, epsilon::Float64)
    T = eltype(x)
    h = handle(x)
    y = similar(x)
    xdesc = tensor_desc(x)
    ydesc = tensor_desc(y)
    sbmvdesc = tensor_desc(scale)

    cudnnBatchNormalizationForwardInference(h, mode, T[1], T[0], xdesc, x,
        ydesc, y, sbmvdesc, scale, bias, estimated_mean, estimated_var, Cdouble(epsilon))

    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(ydesc)
    cudnnDestroyTensorDescriptor(sbmvdesc)
    y
end

function batchnorm_training(mode, x, scale, bias, averagefactor::Float64, epsilon::Float64)
    T = eltype(x)
    h = handle(x)
    y = similar(x)
    xdesc = tensor_desc(x)
    ydesc = tensor_desc(y)
    sbmvdesc = tensor_desc(scale)
    running_mean = zeros(scale)
    running_var = zeros(scale)
    save_mean = similar(scale)
    save_invvar = similar(scale)

    cudnnBatchNormalizationForwardTraining(h, mode, T[1], T[0], xdesc, x, ydesc, y,
    sbmvdesc, scale, bias, Cdouble(averagefactor), running_mean, running_var,
	Cdouble(epsilon), save_mean, save_invvar)

    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(ydesc)
    cudnnDestroyTensorDescriptor(sbmvdesc)
    y, running_mean, running_var, save_mean, save_invvar
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

    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(dydesc)
    cudnnDestroyTensorDescriptor(dxdesc)
    cudnnDestroyTensorDescriptor(sbdesc)
end
