export PTBLM
module PTBLM

function traindata(dir::String=".data/PTBLM")
    fetchdata(dir, "ptb.train.txt")
end

function testdata(dir::String=".data/PTBLM")
    fetchdata(dir, "ptb.test.txt")
end

function fetchdata(dir::String, filename::String)
    mkpath(dir)
    path = joinpath(dir, filename)
    if !isfile(path)
        println("Downloading $path...")
        download("https://raw.githubusercontent.com/tomsercu/lstm/master/data/$filename", path)
    end
    map(open(readlines,path)) do line
        Vector{String}(split(chomp(line)))
    end
end

end
