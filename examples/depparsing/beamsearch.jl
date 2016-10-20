function sequence{T}(state::T)::Vector{T}
    seq = T[]
    s = state
    while s.step != 1
        unshift!(seq, n)
        n = n.prev
    end
    unshift!(seq, n)
    seq
end

"""
    beamsearch

* step::Int
* score::Float64
* prev::T
"""
function beamsearch{T}(initstate::T, beamsize::Int, next::Function)
    lessthan{T}(x::T, y::T) = x.score > y.score
    chart = Vector{T}[]
    push!(chart, [initstate])

    k = 1
    while k <= length(chart)
        prevs = chart[k]
        length(prevs) > beamsize && sort!(prevs, lt=lessthan)
        for i = 1:min(beamsize,length(prevs))
            for s::T in next(prevs[i])
                while s.step > length(chart)
                    push!(chart, T[])
                end
                push!(chart[s.step], s)
            end
        end
        k += 1
    end
    sort!(chart[end], lt=lessthan)
    chart[end][1]
end

function beamsearch2{T}(initstate::T, beamsize::Int, next::Function, getscore::Function)
    chart = Vector{Node{T}}[]
    push!(chart, [Node(initstate)])

    k = 1
    while k <= length(chart)
        nodes = chart[k]
        length(nodes) > beamsize && sort!(nodes, lt=lessthan)
        for i = 1:min(beamsize,length(nodes))
            for s::T in next(nodes[i].state)
                while s.step > length(chart)
                    push!(chart, Node{T}[])
                end
                score = getscore(s) + nodes[i].score
                push!(chart[s.step], Node(s,score,nodes[i]))
            end
        end
        k += 1
    end
    sort!(chart[end], lt=lessthan)
    chart[end][1]
end

"""
    max_violation!

Ref: L. Huang et al, "Structured Perceptron with Inexact Seatch", ACL 2012.
"""
function max_violation!{T}(gold::T, pred::T, train_gold, train_pred)
    goldseq, predseq = to_seq(gold), to_seq(pred)
    maxk, maxv = 1, 0.0
    for k = 1:length(goldseq)
        v = predseq[k].score - goldseq[k].score
        if k == 1 || v >= maxv
            maxk = k
            maxv = v
        end
    end
    for k = 2:maxk
        train_gold(goldseq[k])
        train_pred(predseq[k])
    end
end
