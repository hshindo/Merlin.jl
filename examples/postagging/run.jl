push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../../.."))
push!(LOAD_PATH, dirname(@__FILE__))
ENV["USE_CUDA"] = true
workspace()
using Merlin

a = Merlin.malloc(Array{Float32}, 10, 5)
b = Merlin.malloc(CudaArray{Float32}, (10, 5))

function aaa()
  for i = 1:1000
    a = Merlin.alloc_cpu(Float32, 100, 100)
    #a = Array(Float32,100,100)
    #p = Libc.malloc(4*100*100)
    #p = convert(Ptr{Float32}, p)
    #a = pointer_to_array(p, (50,50))
    #finalizer(a, release)
    i % 100 == 0 && gc()
  end
end

aaa()
gc()
ptrs
a = pointer_to_array(ptrs[1], (100,100))

for p in ptrs
  a = pointer_to_array(p, (100,100))
  fill!(a, 0.0)
end


include("token.jl")
include("model.jl")
include("train.jl")
path = "C:/Users/hshindo/Dropbox/tagging"

@time train(path)
