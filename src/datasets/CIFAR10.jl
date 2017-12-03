module CIFAR10

import ..Datasets.unpack

function getdata(dir::String)
    mkpath(dir)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    println("Downloading $url...")
    path = download(url)
    run(unpack(path,dir,".gz",".tar"))
end

function readdata(data::Vector{UInt8})
    n = Int(length(data)/3073)
    x = Matrix{Float64}(3072, n)
    y = Vector{Int}(n)
    for i = 1:n
        k = (i-1) * 3073 + 1
        y[i] = Int(data[k])
        x[:,i] = data[k+1:k+3072] / 255
    end
    x = reshape(x, 32, 32, 3, n)
    x, y
end

function traindata(dir::String)
    files = [joinpath(dir,"cifar-10-batches-bin","data_batch_$i.bin") for i=1:5]
    all(isfile, files) || getdata(dir)
    data = UInt8[]
    for file in files
        append!(data, open(read,file))
    end
    readdata(data)
end

function testdata(dir::String)
    file = joinpath(dir,"cifar-10-batches-bin","test_batch.bin")
    isfile(file) || getdata(dir)
    readdata(open(read,file))
end

end
