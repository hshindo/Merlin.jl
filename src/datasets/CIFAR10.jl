export CIFAR10
module CIFAR10

using BinDeps

function fetchdata(dir::String)
    mkpath(dir)
    path = joinpath(dir, "cifar-10-binary.tar.gz")
    if !ispath(path)
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
        println("Downloading $url...")
        download(url, path)
    end
    path = joinpath(dir, "cifar-10-batches-bin")
    if !isdir(path)
        run(BinDeps.unpack_cmd(path,dir,".gz",".tar"))
    end
end

function format(data::Vector{UInt8})
    n = Int(length(data)/3073)
    x = Matrix{Float64}(undef, 3072, n)
    y = Vector{Int}(undef, n)
    for i = 1:n
        k = (i-1) * 3073 + 1
        y[i] = Int(data[k])
        x[:,i] = data[k+1:k+3072] / 255
    end
    x = reshape(x, 32, 32, 3, n)
    x, y
end

function traindata(dir::String=".data/CIFAT10")
    fetchdata(dir)
    files = [joinpath(dir,"cifar-10-batches-bin","data_batch_$i.bin") for i=1:5]
    data = UInt8[]
    for file in files
        append!(data, open(read,file))
    end
    format(data)
end

function testdata(dir::String=".data/CIFAT10")
    fetchdata(dir)
    file = joinpath(dir,"cifar-10-batches-bin","test_batch.bin")
    format(open(read,file))
end

end
