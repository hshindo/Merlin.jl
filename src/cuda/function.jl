export @nvrtc

type CuFunction
    m::CuModule # avoid CuModule gc-ed
    ptr::Ptr{Void}
end

function CuFunction(m::CuModule, name::String)
    p = CUfunction[0]
    cuModuleGetFunction(p, m, name)
    CuFunction(m, p[1])
end

Base.unsafe_convert(::Type{Ptr{Void}}, f::CuFunction) = f.ptr

box(x) = pointer_from_objref(x)

macro nvrtc(expr)
    expr.head == :string || throw("expr is not string")
    idx = findfirst(expr.args) do a
        typeof(a) <: String && match(r"__global__ void", a) != nothing
    end
    idx == 0 && throw("Cannot find \"__global__ void\".")
    dict = Dict()
    for i = idx+1:length(expr.args)
        arg = expr.args[i]
        typeof(arg) == Symbol && (dict[arg] = arg)
        typeof(arg) == String && contains(arg, ")") && break
    end
    syms = Expr(:tuple, keys(dict)...)
    dict = Dict()

    quote
        local dict = $dict
        local key = $(esc(syms))
        if haskey(dict, key)
            dict[key]
        else
            local code = $(esc(expr))
            local ptx = NVRTC.compile(code)
            f = load_ptx(ptx)
            dict[key] = f
            f
        end
    end
end

function load_ptx(ptx::String)
    p = Ptr{Void}[0]
    cuModuleLoadData(p, pointer(ptx))
    mod = CuModule(p[1])
    # TODO: multi-device
    for line in split(ptx,'\n')
        m = match(r".visible .entry (.+)\(", line) # find function name
        m == nothing && continue
        fname = Symbol(m[1])
        return CuFunction(mod, string(fname))
    end
end

function (f::CuFunction)(args...;
    dx=1, dy=1, dz=1, bx=128, by=1, bz=1, sharedmem=0, stream=C_NULL)

    argptrs = Ptr{Void}[box(a) for a in args]
    gx = ceil(dx / bx)
    gy = ceil(dy / by)
    gz = ceil(dz / bz)
    cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sharedmem, stream, argptrs, C_NULL)
end
