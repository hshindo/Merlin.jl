type VarArray{T,N}
    data::Vector{T}
    dims
    perm
end

function slicecat{T}(xs::Vector{T}, dim::Int)
    perm = sortperm(xs, by=x->size(x,dim), rev=true)
    xs = map(p -> xs[p], perm)
    dims = Int[size(xs[1])...]
    ys = Array{T}(size(xs[1],))
    for i = 1:size(xs[1],dim)
        dims[dim] = i
        subs = [view(x,dims...), for x in xs]
        ys[i] = cat(ndims(xs[1])+1, subs...)
    end
    ys
end

function batch(xs::Vector{Var})

end
