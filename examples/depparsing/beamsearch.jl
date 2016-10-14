immutable Node{T}
    state::T
    score::Float64
    prev::Node{T}

    Node(state) = new(state, 0.0)
end

lessthan{T}(x::Node{T}, y::Node{T}) = x.score > y.score

function getseq{T}(finalstate::T)
    seq = T[]
    s = finalstate
    while true
        unshift!(seq, s)
        isdefined(s,:prev) || break
        s = s.prev
    end
    unshift!(seq, s)
    seq
end

"""
* next: state -> score, state
"""
function beamsearch{T}(initstate::T, beamsize::Int)
    chart = Vector{T}[]
    push!(chart, [initstate])

    k = 1
    while k <= length(chart)
        states = chart[k]
        length(states) > beamsize && sort!(states, lt=lessthan)
        for i = 1:min(beamsize, length(states))
            for (s,score) in next(states[i])
                while s.step > length(chart)
                    push!(chart, T[])
                end
                push!(chart[s.step], s)
            end
        end
        k += 1
    end

    state = initstate
    k = 1
    while true
        nexts = next(state)

    end
end

function beamsearch{T}(initstate::T, beamsize::Int)
    chart = Vector{T}[]
    push!(chart, [initstate])

    k = 1
    while k <= length(chart)
        states = chart[k]
        length(states) > beamsize && sort!(states, lt=lessthan)
        for i = 1:min(beamsize, length(states))
            for (s,score) in next(states[i])
                while s.step > length(chart)
                    push!(chart, T[])
                end
                push!(chart[s.step], s)
            end
        end
        k += 1
    end
    sort!(chart[end], lt=lessthan)
    chart
end
