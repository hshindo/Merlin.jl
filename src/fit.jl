export minimize!, evaluate

function minimize!(f, dataset, opt; batchsize::Int, shuffle::Bool, device)
    dataset = todevice(dataset, device)
    params = parameters(f)
    loss = 0.0
	n = length(dataset)
    perm = shuffle ? randperm(n) : collect(1:n)
	prog = Progress(n)
	for i = 1:batchsize:n
		j = min(i+batchsize-1, n)
		batch = dataset[perm[i:j]]
		out = f(batch)
		loss += sum(Array(out.data))
        gradient!(out)
		opt.(params)
		update!(prog, j)
	end
    loss
end

function evaluate(f, dataset; batchsize::Int, device)
	dataset = todevice(dataset, device)
    outs = []
	n = length(dataset)
	perm = collect(1:n)
	for i = 1:batchsize:n
		j = min(i+batchsize-1, n)
		batch = dataset[perm[i:j]]
		out = f(batch)
		push!(outs, out)
	end
	outs
end
