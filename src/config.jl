mutable struct Config
    training::Bool
    device::Int
end
const CONFIGS = [Config(true,-1) for i=1:nthreads()]

getconfig() = CONFIGS[threadid()]

istraining() = getconfig().training
settraining(b::Bool) = getconfig().training = b

getdevice() = getconfig().device
function setdevice(dev::Int)
    getconfig().device = dev
    dev >= 0 && CUDA.setdevice(dev)
end
