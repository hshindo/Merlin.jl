export gemm_batch

doc"""
    gemv(tA::Char, alpha, A::Var, x::Var)

* tA: 'T' (transpose) or 'N' (not transpose)

```math
y = \alpha \times \textrm{tA}(A) \times x
```

```julia
T = Float32
A = Var(rand(T,10,5))
x = Var(rand(T,5))
B = gemv('N', 1, A, x)
```
"""
function gemv(tA::Char, alpha::Number, A::Var, x::Var)
    T = eltype(A)
    ydata = gemv(tA, T(alpha), A.data, x.data)
    Var(ydata, ∇gemv!, (tA,alpha,A,x))
end
gemv(tA, alpha, A::Node, x) = Node(gemv, (tA,alpha,A,x))

function ∇gemv!(y::Var, tA::Char, alpha::Number, A::Var, x::Var)
    T = eltype(A)
    if !isnothing(A.grad)
        gy = reshape(y.grad, length(y.grad), 1)
        xdata = reshape(x.data, length(x.data), 1)
        tA == 'N' ?
        gemm!('N', 'T', T(alpha), gy, xdata, T(1), A.grad) :
        gemm!('N', 'T', T(alpha), xdata, gy, T(1), A.grad)
    end
    if !isnothing(x.grad)
        gemv!(tA=='N' ? 'T' : 'N', T(alpha), A.data, y.grad, T(1), x.grad)
    end
end

doc"""
    gemm(tA::Char, tB::Char, alpha, A::Var, B::Var)
    gemm(A::Var, B::Var, [tA='N'], [tB='N'], [alpha=1])

* tA, tB: 'T' (transpose) or 'N' (not transpose)

```math
C = \alpha \times \textrm{tA}(A) \times \textrm{tB}(B)
```

```julia
T = Float32
A = Var(rand(T,10,5))
B = Var(rand(T,10,7))
C = gemm('T', 'N', 1, A, B)
```
"""
function gemm(tA::Char, tB::Char, alpha::Number, A::Var, B::Var)
    T = eltype(A)
    ydata = gemm(tA, tB, T(alpha), A.data, B.data)
    Var(ydata, ∇gemm!, (tA,tB,alpha,A,B))
end
gemm(tA, tB, alpha, A::Node, B) = Node(gemm, (tA,tB,alpha,A,B))
gemm(tA, tB, alpha, A, B::Node) = Node(gemm, (tA,tB,alpha,A,B))

function ∇gemm!(C::Var, tA::Char, tB::Char, alpha::Number, A::Var, B::Var)
    isa(A.grad,Nothing) || ∇gemm_A!(C.grad, tA, tB, alpha, A.grad, B.data)
    isa(B.grad,Nothing) || ∇gemm_B!(C.grad, tA, tB, alpha, A.data, B.grad)
end

function ∇gemm_A!(gC, tA::Char, tB::Char, alpha, gA, B)
    T = eltype(gC)
    if tA == 'N'
        gemm!('N', tB=='N' ? 'T' : 'N', T(alpha), gC, B, T(1), gA)
    else
        gemm!(tB, 'T', T(alpha), B, gC, T(1), gA)
    end
end

function ∇gemm_B!(gC, tA::Char, tB::Char, alpha, A, gB)
    T = eltype(gC)
    if tB == 'N'
        gemm!(tA=='N' ? 'T' : 'N', 'N', T(alpha), A, gC, T(1), gB)
    else
        gemm!('T', tA, T(alpha), gC, A, T(1), gB)
    end
end

doc"""
    gemm_batch(tA::Char, tB::Char, alpha, A::Var, B::Var)
"""
function gemm_batch(tA::Char, tB::Char, alpha, A::Var, B::Var)
    T = eltype(A)
    ydata = gemm_batch(tA, tB, T(alpha), A.data, B.data)
    Var(ydata, ∇gemm_batch!, (tA,tB,alpha,A,B))
end
gemm_batch(A, B; tA='N', tB='N', alpha=1) = gemm_batch(tA, tB, alpha, A, B)

function gemm_batch(tA::Char, tB::Char, alpha, A::Array{T,3}, B::Array{T,3}) where T
    @assert size(A,3) == size(B,3)
    m = size(A, tA == 'N' ? 1 : 2)
    n = size(B, tB == 'N' ? 2 : 1)
    C = Array{T}(m, n, size(A,3))
    for i = 1:size(A,3)
        gemm!(tA, tB, T(alpha), view(A,:,:,i), view(B,:,:,i), T(0), view(C,:,:,i))
    end
    C
end

function gemm_batch(tA::Char, tB::Char, alpha, A::CuArray{T,3}, B::CuArray{T,3}) where T
    CUBLAS.gemm_batched(tA, tB, T(alpha), A, B)
end

function addgrad!(C::Var, ::typeof(gemm_batch), tA::Char, tB::Char, alpha, A::Var, B::Var)
    isvoid(A.grad) || ∇gemm_batch_A!(C.grad, tA, tB, alpha, A.grad, B.data)
    isvoid(B.grad) || ∇gemm_batch_B!(C.grad, tA, tB, alpha, A.data, B.grad)
end

function ∇gemm_batch_A!(gC::Array{T,3}, tA::Char, tB::Char, alpha, gA::Array{T,3}, B::Array{T,3}) where T
    for i = 1:size(gC,3)
        ∇gemm_A!(view(gC,:,:,i), tA, tB, alpha, view(gA,:,:,i), view(B,:,:,i))
    end
end
function ∇gemm_batch_B!(gC::Array{T,3}, tA::Char, tB::Char, alpha, A::Array{T,3}, gB::Array{T,3}) where T
    for i = 1:size(gC,3)
        ∇gemm_B!(view(gC,:,:,i), tA, tB, alpha, view(A,:,:,i), view(gB,:,:,i))
    end
end

function ∇gemm_batch_A!(gC::CuArray{T,3}, tA::Char, tB::Char, alpha, gA::CuArray{T,3}, B::CuArray{T,3}) where T
    if tA == 'N'
        CUBLAS.gemm_batched!('N', tB=='N' ? 'T' : 'N', T(alpha), gC, B, T(1), gA)
    else
        CUBLAS.gemm_batched!(tB, 'T', T(alpha), B, gC, T(1), gA)
    end
end
function ∇gemm_batch_B!(gC::CuArray{T,3}, tA::Char, tB::Char, alpha, A::CuArray{T,3}, gB::CuArray{T,3}) where T
    if tB == 'N'
        CUBLAS.gemm_batched!(tA=='N' ? 'T' : 'N', 'N', T(alpha), A, gC, T(1), gB)
    else
        CUBLAS.gemm_batched!('T', tA, T(alpha), gC, A, T(1), gB)
    end
end
