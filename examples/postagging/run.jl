github_path = "C:/Users/shindo/Documents/GitHub"
push!(LOAD_PATH, github_path)
push!(LOAD_PATH, "$(github_path)/Merlin.jl/examples/postagging")
push!(LOAD_PATH, "/Users/hshindo/.julia/v0.4/Merlin.jl/examples/postagging")
workspace()
using Merlin
using POSTagging
path = "/Users/hshindo/Dropbox/tagging"

function bench()
  a = rand(Float32, 100, 100) |> Variable
  f = ReLU()
  for i = 1:100000
    f(a)
  end
end

@time bench()

@time POSTagging.train(path)
