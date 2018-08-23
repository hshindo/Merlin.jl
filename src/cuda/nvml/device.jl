export free_device, free_devices

function gethandle(dev::Int)
    ref = Ref{Ptr{Cvoid}}()
    @nvml :nvmlDeviceGetHandleByIndex (Cuint,Ptr{Ptr{Cvoid}}) dev ref
    ref[]
end

function getcount()
    ref = Ref{Cuint}()
    @nvml :nvmlDeviceGetCount (Ptr{Cuint},) ref
    Int(ref[])
end

"""
    free_devices(maxcount::Int)

Returns GPU device ids with no running processes
"""
function free_devices(maxcount::Int)
    devs = Int[]
    for i = 0:ndevices()-1
        length(devs) >= maxcount && break
        h = gethandle(i)
        res = @nvml_nocheck(:nvmlDeviceGetComputeRunningProcesses,
            (Ptr{Cvoid},Ptr{Cuint},Ptr{Cvoid}),
            h, Ref{Cuint}(0), C_NULL)
        (res == 0 || res == 7) && push!(devs,i)
    end
    devs
end
free_devices() = free_devices(ndevices())
function free_device()
    devs = free_devices(1)
    isempty(devs) && throw("No free CUDA device.")
    devs[1]
end
