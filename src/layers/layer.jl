abstract Functor

Base.getindex(f::Functor, key::Int) = f.xs[key]
Base.setindex!(f::Functor, value, key::Int) = f.xs[key] = value

function topsort(top::Functor)
    sorted = Layer[]
    dict = ObjectIdDict()
    function visit(l::Layer)
        haskey(dict,l) && return
        dict[l] = l
        for x in l.xs
            visit(x)
        end
        push!(sorted, l)
    end
    visit(top)
    sorted
end


abstract Layer

Base.getindex(l::Layer, key::Int) = l.xs[key]
Base.setindex!(l::Layer, value, key::Int) = l.xs[key] = value

function topsort(top::Layer)
    sorted = Layer[]
    dict = ObjectIdDict()
    function visit(l::Layer)
        haskey(dict,l) && return
        dict[l] = l
        for x in l.xs
            visit(x)
        end
        push!(sorted, l)
    end
    visit(top)
    sorted
end
