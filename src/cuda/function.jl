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

box(x) = x
box(x::Int) = Cint(x)
box(t::NTuple{2,Int}) = Cint2(t[1],t[2])
box(t::NTuple{3,Int}) = Cint3(t[1],t[2],t[3])
box(t::NTuple{4,Int}) = Cint4(t[1],t[2],t[3],t[4])

function (f::CuFunction)(args...;
    dx=1, dy=1, dz=1, bx=128, by=1, bz=1, sharedmem=0, stream=C_NULL)
    argptrs = Ptr{Void}[pointer_from_objref(box(a)) for a in args]
    gx = ceil(dx / bx)
    gy = ceil(dy / by)
    gz = ceil(dz / bz)
    cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sharedmem, stream, argptrs, stream)
end

const ntuple_h = """
template<int N>
struct NTuple {
    const int data[N];
public:
    __device__ int& operator[](const int idx) { return data[idx]; }
};
"""

const array_h = """
template<typename T, int N>
struct Array {
    T *data;
    const int dims[N];
public:
    __device__ int length() {
        int n = dims[0];
        for (int i = 1; i < N; i++) n *= dims[i];
        return n;
    }
    __device__ T& operator[](const int idx) { return data[idx]; }
    __device__ T& operator()(int idx0, int idx1) {
        return data[idx0 + idx1*dims[0]];
    }
    __device__ T& operator()(int idx0, int idx1, int idx2) {
        return data[idx0 + idx1*dims[0] + idx2*dims[0]*dims[1]];
    }
    __device__ T& operator()(int idx0, int idx1, int idx2, int idx3) {
        return data[idx0 + idx1*dims[0] + idx2*dims[0]*dims[1] + idx3*dims[0]*dims[1]*dims[2]];
    }
    __device__ void idx2sub2(const int idx, const int *cumdims, int *subs) {
        int temp = idx;
        for (int i = N-1; i >= 1; i--) {
            int k = temp / cumdims[i];
            subs[i] = k;
            temp -= k * cumdims[i];
        }
        subs[0] = temp;
        return;
    }
    __device__ void idx2sub(const int idx, int *subs) {
        int cumdims[N];
        cumdims[0] = 1;
        for (int i = 1; i < N; i++) cumdims[i] = cumdims[i-1] * dims[i-1];

        int temp = idx;
        for (int i = N-1; i >= 1; i--) {
            int k = temp / cumdims[i];
            subs[i] = k;
            temp -= k * cumdims[i];
        }
        subs[0] = temp;
        return;
    }
    __device__ T& operator()(int *subs) {
        int idx = 0;
        int stride = 1;
        for (int i = 0; i < N; i++) {
            if (dims[i] > 1) idx += subs[i] * stride;
            stride *= dims[i];
        }
        return data[idx];
    }
};
"""
