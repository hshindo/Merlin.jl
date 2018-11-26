export DataLoader
using Random
using ProgressMeter

mutable struct DataLoader
    data
    batchsize::Int
    shuffle::Bool
	device::Int
end

function DataLoader(data; batchsize, shuffle=false, device)
	data = todevice(data, device)
    DataLoader(data, batchsize, shuffle, device)
end

Base.length(loader::DataLoader) = length(loader.data)

function Base.foreach(f, loader::DataLoader)
	perm = loader.shuffle ? randperm(length(loader)) : collect(1:length(loader))
	prog = Progress(length(loader))
	for i = 1:loader.batchsize:length(loader)
		j = min(i+loader.batchsize-1, length(loader))
		data = loader.data[perm[i:j]]
		f(data)
		update!(prog, j)
	end
end

function Base.iterate(loader::DataLoader)
	perm = loader.shuffle ? randperm(length(loader)) : collect(1:length(loader))
	prog = Progress(length(loader))
	iterate(loader, (perm,prog,1))
end

function Base.iterate(loader::DataLoader, state)
	perm, prog, i = state
	if i > length(loader)
		nothing
	else
		j = min(i+loader.batchsize-1, length(loader))
		update!(prog, j)
		data = loader.data[perm[i:j]]
		data, (perm,prog,j+1)
	end
end
