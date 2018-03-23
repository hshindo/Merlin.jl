# If mode in rnnDesc was set to CUDNN_LSTM values of 0, 1, 2 and 3 reference bias
# applied to the input from the previous layer, value of 4, 5, 6 and 7 reference bias
# applied to the recurrent input.
# ‣ Values 0 and 4 reference the input gate.
# ‣ Values 1 and 5 reference the forget gate.
# ‣ Values 2 and 6 reference the new memory gate.
# ‣ Values 3 and 7 reference the output gate.
function setcuda!(lstm::LSTM)
    isa(lstm.w.data,CuArray) && return
    param = eltype(lstm.ws[1])[]
    hx = eltype(lstm.ws[1])[]
    cx = eltype(lstm.ws[1])[]
    coef = lstm.bidirectional ? 2 : 1
    for l = 1:lstm.nlayers
        for d = 1:coef
            i = (l-1)*coef + d
            w = lstm.ws[i].data
            n = l == 1 ? lstm.insize : lstm.hsize*coef
            append!(param, w[1:n,:])
            append!(param, w[n+1:end,:])
            append!(hx, lstm.h0s[i].data)
            append!(cx, lstm.c0s[i].data)
        end
    end
    for l = 1:lstm.nlayers
        for d = 1:coef
            i = (l-1)*coef + d
            b = lstm.bs[i].data
            append!(param, b)
            append!(param, zeros(b)) # CUDNN requires bias for U
        end
    end
    w = CuArray(param)
    lstm.params = (w,)
end
