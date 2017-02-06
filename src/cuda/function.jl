type CuModule
    ptr::Ptr{Void}

    function CuModule(ptr)
        m = new(ptr)
        finalizer(m, cuModuleUnload)
        m
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, m::CuModule) = m.ptr

function CuModule(image::Vector{UInt8})
    p = Ptr{Void}[0]
    cuModuleLoadData(p, image)
    #cuModuleLoadDataEx(p, image, 0, CUjit_option[], Ptr{Void}[])
    CuModule(p[1])
end

type CuFunction
    m::CuModule # avoid CuModule gc-ed
    ptr::Ptr{Void}
end

Base.unsafe_convert(::Type{Ptr{Void}}, f::CuFunction) = f.ptr

function CuFunction(m::CuModule, name::String)
    p = CUfunction[0]
    cuModuleGetFunction(p, m, name)
    CuFunction(m, p[1])
end

function CuFunction(code::String)
    code = replace(code, "Float32", "float")
    code = replace(code, "Int", "int")

    ptx = NVRTC.compile(code)
    p = Ptr{Void}[0]
    cuModuleLoadData(p, pointer(ptx))
    mod = CuModule(p[1])

    fnames = []
    for line in split(ptx,'\n')
        m = match(r".visible .entry (.+)\(", line) # find function name
        m == nothing && continue
        push!(fnames, String(m[1]))
    end
    length(fnames) > 1 && throw("Multiple functions are found.")
    CuFunction(mod, fnames[1])
end

macro compile(expr)
    expr.head == :string || throw("expr is not string")
    idx = findfirst(expr.args) do a
        isa(a,String) && match(r"__global__ void", a) != nothing
    end
    idx == 0 && throw("Cannot find \"__global__ void\".")
    dict = Dict()
    for i = idx+1:length(expr.args)
        arg = expr.args[i]
        isa(arg, Symbol) && (dict[arg] = arg)
        isa(arg, String) && contains(arg, ")") && break # end of function declaration
    end
    syms = Expr(:tuple, keys(dict)...)
    dict = Dict()

    quote
        local dict = $dict
        local key = $(esc(syms))
        get!(dict, key) do
            local code = $(esc(expr))
            local ptx = NVRTC.compile(code)
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
    end
end

immutable Cint2
    i1::Cint
    i2::Cint
end
immutable Cint3
    i1::Cint
    i2::Cint
    i3::Cint
end
immutable Cint4
    i1::Cint
    i2::Cint
    i3::Cint
    i4::Cint
end

box(x) = pointer_from_objref(x)
box(x::Int) = pointer_from_objref(Cint(x))
box(x::Float32) = pointer_from_objref(x)
box(t::NTuple{2,Int}) = pointer_from_objref(Cint2(t[1],t[2]))
box(t::NTuple{3,Int}) = pointer_from_objref(Cint3(t[1],t[2],t[3]))
box(t::NTuple{4,Int}) = pointer_from_objref(Cint4(t[1],t[2],t[3],t[4]))

function aaa_h(x::NTuple)
    """
    template<int N, typename T>
    struct NTuple {
        const T data[N];
    public:
        __device__ T& operator[](const int idx) { return data[idx]; }
    };
    """
end

function (f::CuFunction)(args...;
    dx=1, dy=1, dz=1, bx=128, by=1, bz=1, sharedmem=0, stream=C_NULL)

    for arg in args
        aaa(arg)
        b = cubox(arg)
        p = pointer_from_objref(cubox(arg))
    end

    argptrs = Ptr{Void}[box(a) for a in args]
    gx = ceil(dx / bx)
    gy = ceil(dy / by)
    gz = ceil(dz / bz)
    cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sharedmem, stream, argptrs, stream)
end
