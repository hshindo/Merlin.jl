type MsgPackSerializer
    data
end

to_msgpack(x::Array) = vec(x)
