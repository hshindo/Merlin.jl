using Merlin
using Base.Threads
using LibCUDA

function bench()
    results = Array{Any}(2)
    @threads for dev = 0:1
        T = Float32
        x = rand(T,100,100)
        f = Linear(T,100,100)
        y = f(Var(x))
        results[dev+1] = Array(y.data)
    end
end

@time bench()
#@threads for i = 1:2
    # a[i] = threadid()
#    setdevice(0)
#    a[i] = threadid()
#end
