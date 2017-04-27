function forward{T}(::typeof(batchnorm), x::CuArray{T}, scale::CuArray, bias,
    mean, var; epsilon=0.001)

    h = CUDNN.handle(x)
    y = similar(x)
    xdesc = CUDNN.TensorDesc(x)
    sbmvdesc = CUDNN.TensorDesc(scale)
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

function batchnorm_training(mode, x, scale, bias, averagefactor::Float64, epsilon::Float64)
    T = eltype(x)
    h = handle(x)
    y = similar(x)
    xdesc = TensorDesc(x)
    ydesc = TensorDesc(y)
    sbmvdesc = TensorDesc(scale)
    running_mean = zeros(scale)
    running_var = zeros(scale)
    save_mean = similar(scale)
    save_invvar = similar(scale)

    cudnnBatchNormalizationForwardTraining(h, mode, T[1], T[0], xdesc, x, ydesc, y,
        sbmvdesc, scale, bias, Cdouble(averagefactor), running_mean, running_var,
	    Cdouble(epsilon), save_mean, save_invvar)

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
end
