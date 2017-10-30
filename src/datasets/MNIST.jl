module MNIST

using GZip

function traindata(path::String)
    x = gzread(path,"train-images-idx3-ubyte.gz")[17:end]
    x = reshape(x/255, 28*28, 60000)
    x = Matrix{Float32}(x)
    y = gzread(path,"train-labels-idx1-ubyte.gz")[9:end]
    y = Vector{Int}(y)
    x, y
end

function testdata(path::String)
    data = gzread(path,"t10k-images-idx3-ubyte.gz")[17:end]
    x = reshape(data/255, 28*28, 10000)
    x = Matrix{Float32}(x)
    data = gzread(path,"t10k-labels-idx1-ubyte.gz")[9:end]
    y = Vector{Int}(data)
    x, y
end

function gzread(path::String, filename::String)
    mkpath(path)
    path = joinpath(path, filename)
    if !isfile(path)
        println("Downloading $filename...")
        download("http://yann.lecun.com/exdb/mnist/$filename", path)
    end
    gzopen(read, path)
end

end
