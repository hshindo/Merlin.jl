export Sequence

type Sequence
    funs::Vector
end

Sequence(funs...) = Sequence([funs...])

@compat function (seq::Sequence)(x)
    for f in seq.funs
        x = f(x)
    end
    x
end
