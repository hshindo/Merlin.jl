module Actions
    const shift = 1
    const left = 2
    const right = 3
end

type State
    tokens::Vector{Token}
    step::Int
    top::Int
    left::State
    right::Int
    lch::State # left-most child
    rch::State
    score::Float64
    prev::State

    State() = new(Token[], 0)
    State(tokens) = new(tokens, 1, 1, State(), 2, State(), State(), 0.0)
    State(tokens, step, top, left, right, lch, rch) = new(tokens, step, top, left, right, lch, rch, 0.0, left)
end

#State(tokens::Vector{Token}) = State(tokens, 1, 1, State(), 2, State(), State())

function State(s::State, act::Int)
    top, left, right, lch, rch = begin
        if act == Actions.shift
            s.right, s, s.right+1, State(), State()
        elseif act == Actions.left
            s.top, s.left.left, s.right, s.left, s.rch
        elseif act == Actions.right
            s.left.top, s.left.left, s.right, s.left.lch, s
        else
            throw("Invalid action: $(act)")
        end
    end
    State(s.tokens, s.step+1, top, left, right, lch, rch)
end

Base.done(s::State) = s.step == 2 * (length(s.tokens)-1) + 1

Base.isnull(s::State) = s.step == 0

function nextpred(s::State)::Vector{State}
    done(s) && return State[]
    states = State[]
    isnull(s.left) || push!(states, State(s,Actions.left), State(s,Actions.right))
    s.right <= length(s.tokens) && push!(states, State(s,Actions.shift))
    states
end

function nextgold(s::State)::Vector{State}
    done(s) && return State[]
    if isnull(s.left)
        act = Actions.shift
    else
        tokens = s.tokens
        s0, s1 = tokens[s.top], tokens[s.left.top]
        if s1.head == s.top
            act = Actions.left
        elseif s0.head == s.left.top
            reducable = all(i -> tokens[i].head != s.top, s.right:length(tokens))
            act = reducable ? Actions.right : Actions.shift
        elseif s.right <= length(tokens)
            act = Actions.shift
        else
            throw("Invalid data.")
        end
    end
    [State(s,act)]
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
