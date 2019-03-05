export DataLoader
using Random
using ProgressMeter

mutable struct DataLoader
	batchfun
    data::Vector
    batchsize::Int
    shuffle::Bool
	device::Int
end

function DataLoader(batchfun, data::Vector; batchsize, shuffle, device)
    DataLoader(batchfun, data, batchsize, shuffle, device)
end

function Base.length(dl::DataLoader)
	n = length(dl.data) / dl.batchsize
	ceil(Int, n)
end

function Base.iterate(dl::DataLoader)
	perm = dl.shuffle ? randperm(length(dl)) : collect(1:length(dl))
	iterate(dl, (perm,1))
end

function Base.iterate(dl::DataLoader, state)
	perm, i = state
	if i > length(dl)
		nothing
	else
		j = min(i+dl.batchsize-1, length(dl))
		data = dl.data[perm[i:j]]
		data = dl.batchfun(data)
		data, (perm,j+1)
	end
end
