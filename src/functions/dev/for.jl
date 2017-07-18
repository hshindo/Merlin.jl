export For

struct For
    graph
end

function For(f, data)
    g = nothing
    for x in data
        y = f(x)
        println(x)
        g = compile((x,),y)
        break
    end
    _for = For(g)
    Var(nothing, _for, data)
end

function (f::For)(xs::Vector)
    ys = Var[]
    for x in xs
        y = f.graph(x)
        push!(ys, y)
    end
    cat(ndims(ys[1].data), ys...)
end
