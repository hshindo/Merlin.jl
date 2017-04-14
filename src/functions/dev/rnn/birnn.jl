type BiRNN
    f
    b
end

function (f::BiRNN)(x::Var)
    f(x)
end
