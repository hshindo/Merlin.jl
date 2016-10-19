immutable Node{T}
    state::T
    score::Float64
    prev::Node{T}

    Node(state) = new(state, 0.0)
end

lessthan{T}(x::Node{T}, y::Node{T}) = x.score > y.score

function getseq{T}(node::Node{T})
    seq = Node{T}[]
    n = node
    while true
        unshift!(seq, n)
        isdefined(s,:prev) || break
        s = s.prev
    end
    unshift!(seq, s)
    seq
end

"""
    beamsearch
"""
function beamsearch{T}(beamsize::Int, initstate::T, getscore)
    chart = Vector{Node{T}}[]
    push!(chart, [Node(initstate)])

    k = 1
    while k <= length(chart)
        nodes = chart[k]
        length(nodes) > beamsize && sort!(nodes, lt=lessthan)
        for i = 1:min(beamsize,length(nodes))
            for state::T in next(nodes[i])
                while state.step > length(chart)
                    push!(chart, Node{T}[])
                end
                score = getscore(state) + nodes[i].score
                push!(chart[state.step], Node(state,score,nodes[i]))
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
