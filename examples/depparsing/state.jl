const Shift = 1
const ReduceL = 2
const ReduceR = 3

type State
    tokens::Vector{Token}
    step::Int
    top::Int
    right::Int
    left::State
    lc::State
    rc::State

    State(tokens) = new(tokens, 1, 1, 2)
end

Base.done(s::State) = isdefined(s,:left) && s.right == length(s.tokens)+1

function Base.next(s::State, acts::Vector{Int})
    scores = s.scorefun(s, acts)
    states = State[]
    for i = 1:length(acts)
        push!(states, expand(s, acts[i], scores[i]))
    end
    states
end

function Base.next(s::State, act::Int, score::Float64)
    top, lcs, rcs, left, right = begin
        if act == shift
            s.right, Int[], Int[], s, s.right+1
        elseif act == reducel
            _lcs = length(s.lcs) == 0 ? [s.left.top] : [s.left.top,s.lcs[1]]
            s.top, _lcs, s.rcs, s.left.left, s.right
        elseif act == reducer
            _rcs = length(s.left.rcs) == 0 ? [s.top] : [s.top,s.left.rcs[1]]
            s.left.top, s.left.lcs, _rcs, s.left.left, s.right
        else
            throw("Invalid action: $(act)")
        end
    end
    score = s.score + score
    State(top, lcs, rcs, left, right, s.step+1, s, act, s.tokens, s.scorefun, score, nothing)
end

function expand_gold(s::State)
    isfinal(s) && return State[]
    if s.left == nothing
        act = shift
    else
        tokens = s.tokens
        s0, s1 = tokens[s.top], tokens[s.left.top]
        if s1.headid == s.top
            act = reducel
        elseif s0.headid == s.left.top
            reducable = all(i -> tokens[i].headid != s.top, s.right:length(tokens))
            act = reducable ? reducer : shift
        elseif s.right <= length(tokens)
            act = shift
        else
            throw("Invalid")
        end
    end
    expand(s, [act])
end

function expand_pred(s::State)
    isfinal(s) && return State[]
    acts = Int[]
    s.left == nothing || push!(acts, reducel, reducer)
    s.right <= length(s.tokens) && push!(acts, shift)
    expand(s, acts)
end

function toheads(s::State)
    heads = fill(0, length(s.tokens))
    while s != nothing
        length(s.lcs) > 0 && (heads[s.lcs[1]] = s.top)
        length(s.rcs) > 0 && (heads[s.rcs[1]] = s.top)
        s = s.prev
    end
    heads
end
