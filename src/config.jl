export setcpu, setcuda, settrain

mutable struct Config
    device::Int
    train::Bool
end
const CONFIGS = [Config(-1,true) for i=1:nthreads()]

getconfig() = CONFIGS[threadid()]
setconfig(config::Config) = CONFIGS[threadid()] = config

iscpu() = getconfig().device < 0
iscuda() = !iscpu()

getdevice() = getconfig().device
function setdevice(dev::Int)
    getconfig().device = dev
    dev >= 0 && CUDA.setdevice(dev)
end
function setdevice(f::Function, dev::Int)
    _dev = getdevice()
    setdevice(dev)
    f()
    setdevice(_dev)
end
setcpu() = setdevice(-1)
setcpu(f::Function) = setdevoce(f, -1)
function setcuda(dev::Int=-1)
    dev < 0 && (dev = CUDA.free_device())
    setdevice(dev)
end
function setcuda(f::Function, dev::Int=-1)
    CUDA.AVAILABLE || return
    dev < 0 && (dev = CUDA.free_device())
    setdevice(f, dev)
end

istrain() = getconfig().train
settrain(b::Bool) = getconfig().train = b
