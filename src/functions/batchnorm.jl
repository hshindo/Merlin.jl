export batchnorm_training, batchnorm_inference

batchnorm_training(x::CuArray, scale::CuArray, bias::CuArray, mean::CuArray,
    invvar::CuArray, factor, epsilon) = JuCUDNN.batchnorm_training(
    CUDNN_BATCHNORM_SPATIAL, x, scale, bias, factor, mean, invvar, epsilon)

batchnorm_inference(x::CuArray, scale::CuArray, bias::CuArray, mean::CuArray,
    invvar::CuArray, epsilon) = JuCUDNN.batchnorm_inference(
    CUDNN_BATCHNORM_SPATIAL, x, scale, bias, mean, invvar, epsilon)

∇batchnorm!(x::CuArray, gx::CuArray, gy::CuArray, scale::CuArray,
    resultscale::CuArray, resultbias::CuArray, savemean::CuArray,
    saveinvvar::CuArray, epsilon) = JuCUDNN.∇batchnorm!(
    CUDNN_BATCHNORM_SPATIAL, x, gy, scale, resultscale, resultbias, epsilon,
    savemean, saveinvvar, gx)
