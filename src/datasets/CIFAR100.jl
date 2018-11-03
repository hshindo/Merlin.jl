export CIFAR100
module CIFAR100

using BinDeps

function fetchdata(dir::String)
    mkpath(dir)
    path = joinpath(dir, "cifar-100-binary.tar.gz")
    if !ispath(path)
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
        println("Downloading $url...")
        download(url, path)
    end
    path = joinpath(dir, "cifar-100-binary")
    if !isdir(path)
        run(BinDeps.unpack_cmd(path,dir,".gz",".tar"))
    end
end

function format(data::Vector{UInt8})
    n = Int(length(data)/3074)
    x = Matrix{Float64}(undef, 3072, n)
    y = Matrix{Int}(undef, 2, n)
    for i = 1:n
        k = (i-1) * 3074 + 1
        y[:,i] = data[k:k+1]
        x[:,i] = data[k+2:k+3073] / 255
    end
    x = reshape(x, 32, 32, 3, n)
    x, y
end

function traindata(dir::String=".data/CIFAR100")
    fetchdata(dir)
    filepath = joinpath(dir, "cifar-100-binary", "train.bin")
    format(open(read,filepath))
end

function testdata(dir::String=".data/CIFAR100")
    fetchdata(dir)
    filepath = joinpath(dir, "cifar-100-binary", "test.bin")
    format(open(read,filepath))
end

end
