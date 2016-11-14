export Session

type Session
    f
    mp::MemoryPool
end

function Session(dev::String)
end

Session(f) = Session(f, MemoryPool())

function (sess::Session)(x)
    x.sess = sess
    sess.f(x)
end
