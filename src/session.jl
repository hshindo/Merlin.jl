export Session

type Session
    f
    array
    index::Int
end

function Session(f)
    Session(f, nothing, 1)
end

function (sess::Session)(x::Var)
    sess.array = similar(sess.array, 100)
    x.sess = sess
    sess.f(x)
end

function alloc!(sess::Session, T::Type, dims::Tuple)
    len = prod(dims) * sizeof(T)
    @assert len > 0

    count = length(sess.array) - sess.index + 1
    if count < len
        l = length(sess.array)
        while l < sess.index+len-1
            l *= 2
        end
        sess.array = Array(Bool, l)
        sess.index = 1
    end

    p = pointer(sess.array, sess.index)
    sess.index += len
    unsafe_wrap(Array, Ptr{T}(p), dims)
end
