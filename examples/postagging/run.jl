github_path = "C:/Users/shindo/Documents/GitHub"
push!(LOAD_PATH, github_path)
push!(LOAD_PATH, "$(github_path)/Merlin.jl/examples/postagging")
push!(LOAD_PATH, "/Users/hshindo/.julia/v0.4/Merlin.jl/examples/postagging")
workspace()
using Merlin
using CUDArt
using Merlin.CUDNN
using POSTagging
path = "/Users/hshindo/Dropbox/tagging"

dirname(@__FILE__)
r = rand(Float32, 10, 3) - 0.3
x = Var(r)
f = ReLU()
y = forward!(f, x)
fill!(y.grad, 1.0)
fill!(x.grad, 0.0)
x
backward!(f)
x

r = convert(Array{Float32}, randn(10, 5))
x = CudaArray(r)
x = CudaVar(x)
f = ReLU()


x = Var(r)
f = ReLU()
y = forward!(f, x)

function bench()
  #d_A = CudaArray(Float32, (100,200,3,2))
  a = Array(Float32, 100,200,3,2)
  for i = 1:1000
    for j = 1:length(a)
      if a[j] > Float32(0)
        a[j] += a[j]
      end
    end
    #d = CUDNN.create_tensor_descriptor(d_A)
  end
end

@time bench()

@time POSTagging.train(path)
