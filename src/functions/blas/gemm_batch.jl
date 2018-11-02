export gemm_batch

doc"""
    gemm_batch(tA::Char, tB::Char, alpha, A::Var, B::Var)
"""
function gemm_batch(tA::Char, tB::Char, alpha, A::Var, dimsA, B::Var, dimsB)
    T = eltype(A)
    ydata = gemm_batch(tA, tB, T(alpha), A.data, B.data)
    Var(ydata, ∇gemm_batch!, (tA,tB,alpha,A,B))
end
gemm_batch(tA::Char, tb::Char, alpha, A::Node, B) = Node(gemm_batch, (tA,tB,alpha,A,B))
gemm_batch(tA::Char, tb::Char, alpha, A, B::Node) = Node(gemm_batch, (tA,tB,alpha,A,B))
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

function ∇gemm_batch!(C::Var, tA::Char, tB::Char, alpha, A::Var, B::Var)
    isnothing(A.grad) || ∇gemm_batch_A!(C.grad, tA, tB, alpha, A.grad, B.data)
    isnothing(B.grad) || ∇gemm_batch_B!(C.grad, tA, tB, alpha, A.data, B.grad)
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
