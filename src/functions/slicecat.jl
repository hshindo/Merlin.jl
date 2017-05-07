export slicecat

function slicecat(dim::Int, xs::Vector{Var})
    issorted(xs, by=x->length(x.data), rev=true) || throw("xs must be sorted by length.")



    ys = Array{Var}(size(xs[1].data,dim))
    N = ndims(xs[1].data)
    dims = Any[Colon() for i=1:N]
    for i = 1:length(ys)
        dims[N] = i
        subs = []
        for x in xs
            size(x.data,dim) < i && break
            push!(subs, view(x.data,dims...))
        end
        y = cat(N, subs...)
        ys[i] = cat(N, subs...)
    end
    ys
end
