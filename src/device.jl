module Backend



end

export setdevice!

struct CPU
end

struct CUDA
    id::Int
end

doc"""
    setdevice!(x::Var, device)
"""
function setbackend!(x::Var, device)
    if isa(device, CPU)
        if isa(x.data, CuArray)
            x.data = CuArray(x.data)
            isvoid(x.grad) || (x.grad = CuArray(x.grad))
        end
    elseif isa(device, CUDA)
        if isa(x.data, Array)
            x.data = Array(x.data)
            isvoid(x.grad) || (x.grad = Array(x.grad))
        end
    else
        throw("Unknown device: $device")
    end
end

function unify_backend!(x1::Var, x2::Var)
    if isa(x1.data, Array)
        isa(x2.data,CuArray) && CUDA()
    end
end
