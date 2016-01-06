github_path = "C:/Users/shindo/Documents/GitHub"
push!(LOAD_PATH, github_path)
push!(LOAD_PATH, "$(github_path)/Merlin.jl/examples/postagging")
push!(LOAD_PATH, "/Users/hshindo/.julia/v0.4/Merlin.jl/examples/postagging")
workspace()
using Merlin
using POSTagging
path = "C:/Users/shindo/Dropbox/tagging"

ndims(rand(10,2))
v = Var(rand(Float32, 10, 5))
Merlin.default(v,Float64, 3)

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
