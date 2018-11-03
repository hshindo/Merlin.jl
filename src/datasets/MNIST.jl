export MNIST
module MNIST

using CodecZlib

function traindata(dir::String=".data/MNIST")
    x = fetchdata(dir,"train-images-idx3-ubyte.gz")[17:end]
    x = reshape(x/255, 28*28, 60000)
    x = Matrix{Float32}(x)
    y = fetchdata(dir,"train-labels-idx1-ubyte.gz")[9:end]
    y = Vector{Int}(y)
    x, y
end

function testdata(dir::String=".data/MNIST")
    data = fetchdata(dir,"t10k-images-idx3-ubyte.gz")[17:end]
    x = reshape(data/255, 28*28, 10000)
    x = Matrix{Float32}(x)
    data = fetchdata(dir,"t10k-labels-idx1-ubyte.gz")[9:end]
    y = Vector{Int}(data)
    x, y
end

function fetchdata(dir::String, filename::String)
    mkpath(dir)
    path = joinpath(dir, filename)
    if !isfile(path)
        println("Downloading $path...")
        download("http://yann.lecun.com/exdb/mnist/$filename", path)
    end
    open(s -> read(GzipDecompressorStream(s)), path)
end

end
