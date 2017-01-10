GRU_training(x::CuArray, hx::CuArray, cx::CuArray, droprate) =
    JuCUDNN.rnn_training(x, hx, cx, droprate, CUDNN_LINEAR_INPUT,
    CUDNN_UNIDIRECTIONAL, CUDNN_GRU)

GRU_inference(x::CuArray, hx::CuArray, cx::CuArray, w::CuArray, dropdesc) =
    JuCUDNN.rnn_inference(x, hx, cx, w, dropdesc, CUDNN_LINEAR_INPUT,
    CUDNN_UNIDIRECTIONAL, CUDNN_GRU)

∇GRU_data!(x::CuArray, gx::CuArray, hx::CuArray, ghx::CuArray, cx::CuArray,
    gcx::CuArray, w::CuArray, y::CuArray, gy::CuArray, ghy::CuArray,
    gcy::CuArray, dropdesc) = JuCUDNN.∇rnn_data!(
    x, hx, cx, w, y, gy, ghy, gcy, dropdesc, CUDNN_LINEAR_INPUT,
    CUDNN_UNIDIRECTIONAL, CUDNN_GRU, gx, ghx, gcx)

∇GRU_weight!(x::CuArray, hx::CuArray, w::CuArray, gw::CuArray, y::CuArray,
    dropdesc) = JuCUDNN.∇rnn_weight!(x, hx, y, w, dropdesc, CUDNN_LINEAR_INPUT,
    CUDNN_UNIDIRECTIONAL, CUDNN_GRU, gw)
