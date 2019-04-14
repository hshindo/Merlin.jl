export minimize!, evaluate

#=
function minimize!(f, dataloader, opt)
	settraining(true)
    params = parameters(f)
    loss = 0.0
	n = length(dataloader)
    perm = shuffle ? randperm(n) : collect(1:n)
	prog = Progress(n)
	i = 1
	for data in dataloader
		out = f(data)
		loss += sum(Array(out.data))
		getdevice() >= 0 && CUDA.synchronize()
        gradient!(out)
		opt.(params)
		getdevice() >= 0 && CUDA.synchronize()
		update!(prog, i)
		i += 1
	end
    loss
end
=#

batch(xs...) = throw("Batch function not implemented.")

function minimize!(f, dataset, opt; batchsize::Int, shuffle::Bool)
	settraining(true)
    # dataset = todevice(dataset)
    params = parameters(f)
    loss = 0.0
	n = length(dataset)
    perm = shuffle ? randperm(n) : collect(1:n)
	prog = Progress(n)
	for i = 1:batchsize:n
		j = min(i+batchsize-1, n)
		x = batch(dataset, perm[i:j])
		out = f(x)
		loss += sum(Array(out.data))
		getdevice() >= 0 && CUDA.synchronize()
        gradient!(out)
		opt.(params)
		getdevice() >= 0 && CUDA.synchronize()
		update!(prog, j)
	end
    loss
end

function evaluate(f, dataset; batchsize::Int)
	settraining(false)
	# dataset = todevice(dataset)
    outs = []
	n = length(dataset)
	perm = collect(1:n)
	prog = Progress(n)
	for i = 1:batchsize:n
		j = min(i+batchsize-1, n)
		x = batch(dataset, perm[i:j])
		out = f(x)
		push!(outs, out)
		getdevice() >= 0 && CUDA.synchronize()
		update!(prog, j)
	end
	map(identity, outs)
end
