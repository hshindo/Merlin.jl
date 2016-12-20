using Clang

includes = ["/usr/local/include",
            "/usr/include",
            "/usr/lib/gcc/x86_64-linux-gnu/4.8/include",
            "/usr/lib/gcc/x86_64-linux-gnu/4.8/include-fixed"]
headers = ["/usr/local/cuda/include/nvrtc.h"]

# Customize how functions, constants, and structs are written
const skip_expr = []
const skip_error_check = []
function rewriter(ex::Expr)
    if in(ex, skip_expr)
        return :()
    end
    # Empty types get converted to Void
    if ex.head == :type
        a3 = ex.args[3]
        if isempty(a3.args)
            objname = ex.args[2]
            return :(typealias $objname Void)
        end
    end
    ex.head == :function || return ex
    decl, body = ex.args[1], ex.args[2]
    # omit types from function prototypes
    for i = 2:length(decl.args)
        a = decl.args[i]
        if a.head == :(::)
            decl.args[i] = a.args[1]
        end
    end
    # Error-check functions that return a cudaError_t (with some omissions)
    ccallexpr = body.args[1]
    if ccallexpr.head != :ccall
      error("Unexpected body expression: ", body)
    end
    rettype = ccallexpr.args[2]
    if rettype == :nvrtcResult
      fname = decl.args[1]
      if !in(fname, skip_error_check)
        body.args[1] = Expr(:call, :check_nvrtcresult, deepcopy(ccallexpr))
      end
    end
    ex
end

rewriter(A::Array) = [rewriter(a) for a in A]
rewriter(s::Symbol) = string(s)
rewriter(arg) = arg

context = wrap_c.init(output_file = "libnvrtc.jl",
                      common_file = "libnvrtc_types.jl",
                      header_library = _->"libnvrtc",
                      headers = headers,
                      clang_includes = includes,
                      clang_diagnostics = true,
                      header_wrapped=(x,y)->contains(y,"nvrtc"),
                      rewriter = rewriter)

context.options = wrap_c.InternalOptions(true,true)  # wrap structs, too

# Execute the wrap
run(context)
