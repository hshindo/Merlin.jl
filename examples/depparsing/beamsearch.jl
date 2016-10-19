immutable Node{T}
    state::T
    score::Float64
    prev::Node{T}

    Node(state::T) = new(state, 0.0)
    Node(state::T, score, prev) = new(state, score, prev)
end

Node{T}(state::T) = Node{T}(state)
Node{T}(state::T, score, prev) = Node{T}(state, score, prev)

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
function beamsearch{T}(initstate::T, beamsize::Int, getscore)
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
