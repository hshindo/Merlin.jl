export slicecat

function slicecat{T}(xs::Vector{T}, dim::Int)
    perm = sortperm(xs, by=x->size(x,dim), rev=true)
    xs = map(p -> xs[p], perm)
    dims = Any[Colon() for i=1:ndims(T)]
    ys = Array{T}(size(xs[1],dim))
    for i = 1:length(ys)
        dims[dim] = i:i
        subs = []
        for x in xs
            size(x,dim) < i && break
            push!(subs, view(x,dims...))
        end
        ys[i] = cat(ndims(T), subs...)
    end
    ys
end

type Arrays{T,N}
    data::Vector{T}
    inds::Vector{Int}
end

function minibatch(data::Vector{Var}, size::Int)
    for i = 1:size:length(data)
        
    end
end
