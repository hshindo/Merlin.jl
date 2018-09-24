export geam!, gemm_batched!, gemm_batched

for (f,T) in (
    (:(:cublasSgeam),:Float32),
    (:(:cublasDgeam),:Float64))
    @eval begin
        function geam!(tA::Char, tB::Char, alpha::$T, A::CuMatrix{$T},
            beta::$T, B::CuMatrix{$T}, C::CuMatrix{$T})

            m = size(A, tA == 'N' ? 1 : 2)
            n = size(A, tA == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2)
                throw(DimensionMismatch(""))
            end

            @cublas($f, (
                Ptr{Cvoid},Cint,Cint,Cint,Cint,
                Ptr{$T},Ptr{$T},Cint,
                Ptr{$T},Ptr{$T},Cint,
                Ptr{$T},Cint),
                gethandle(), cublasop(tA), cublasop(tB), m, n,
                [alpha], A, stride(A,2),
                [beta], B, stride(B,2),
                C, stride(C,2))
            C
        end
    end
end

function Base.transpose(x::CuMatrix{T}) where T
    t = similar(x, size(x,2), size(x,1))
    geam!('T', 'N', T(1), x, T(0), t, t)
    t
end
Base.transpose(x::CuVector) = transpose(reshape(x,length(x),1))

for (f,T) in (
        (:(:cublasDgemmBatched),:Float64),
        (:(:cublasSgemmBatched),:Float32))
    @eval begin
        function gemm_batched!(tA::Char, tB::Char, m::Int, n::Int, k::Int,
            alpha::$T, As::Vector{Ptr{$T}}, lda::Int, Bs::Vector{Ptr{$T}}, ldb::Int,
            beta::$T, Cs::Vector{Ptr{$T}}, ldc::Int, batchcount::Int)

            if (length(As) != length(Bs) || length(As) != length(Cs))
                throw(DimensionMismatch(""))
            end

            @cublas($f, (
                Ptr{Cvoid},Cint,Cint,Cint,Cint,Cint,
                Ptr{$T},Ptr{Ptr{$T}},Cint,
                Ptr{Ptr{$T}},Cint,
                Ptr{$T},Ptr{Ptr{$T}},Cint,Cint),
                gethandle(), cublasop(tA), cublasop(tB), m, n, k,
                [alpha], CuArray(As), lda, CuArray(Bs), ldb,
                [beta], CuArray(Cs), ldc, batchcount)
            Cs
        end
    end
end

function gemm_batched!(tA::Char, tB::Char,
    alpha::T, As::Vector{CuMatrix{T}}, Bs::Vector{CuMatrix{T}},
    beta::T, Cs::Vector{CuMatrix{T}}) where T

    for (A,B,C) in zip(As,Bs,Cs)
        m = size(A, tA == 'N' ? 1 : 2)
        kA = size(A, tA == 'N' ? 2 : 1)
        kB = size(B, tB == 'N' ? 1 : 2)
        n = size(B, tB == 'N' ? 2 : 1)
        if m != size(C,1) || n != size(C,2) || kA != kB
            throw(DimensionMismatch(""))
        end
    end
    m = size(As[1], tA == 'N' ? 1 : 2)
    k = size(As[1], tA == 'N' ? 2 : 1)
    n = size(Bs[1], tB == 'N' ? 2 : 1)
    ptrAs = map(pointer, As)
    ptrBs = map(pointer, Bs)
    ptrCs = map(pointer, Cs)
    lda = stride(As[1], 2)
    ldb = stride(Bs[1], 2)
    ldc = stride(Cs[1], 2)
    gemm_batched!(tA, tB, m, n, k, alpha, ptrAs, lda, ptrBs, ldb, beta, ptrCs, ldc, length(As))
    Cs
end

function gemm_batched!(tA::Char, tB::Char, alpha::T, A::CuArray{T,3}, B::CuArray{T,3}, beta::T, C::CuArray{T,3}) where T
    @assert size(A,3) == size(B,3) == size(C,3)
    m = size(A, tA == 'N' ? 1 : 2)
    kA = size(A, tA == 'N' ? 2 : 1)
    kB = size(B, tB == 'N' ? 1 : 2)
    n = size(B, tB == 'N' ? 2 : 1)
    if m != size(C,1) || n != size(C,2) || kA != kB
        throw(DimensionMismatch(""))
    end
    k = kA
    ptrAs = [pointer(A,(i-1)*size(A,1)*size(A,2)+1) for i=1:size(A,3)]
    ptrBs = [pointer(B,(i-1)*size(B,1)*size(B,2)+1) for i=1:size(B,3)]
    ptrCs = [pointer(C,(i-1)*size(C,1)*size(C,2)+1) for i=1:size(C,3)]
    lda = stride(A, 2)
    ldb = stride(B, 2)
    ldc = stride(C, 2)
    gemm_batched!(tA, tB, m, n, k, alpha, ptrAs, lda, ptrBs, ldb, beta, ptrCs, ldc, size(A,3))
    C
end

function gemm_batched(tA::Char, tB::Char, alpha::T, As::Vector{CuMatrix{T}}, Bs::Vector{CuMatrix{T}}) where T
    m = size(As[1], tA == 'N' ? 1 : 2)
    n = size(Bs[1], tB == 'N' ? 2 : 1)
    Cs = [CuArray{T}(m,n) for i=1:length(As)]
    gemm_batched!(tA, tB, alpha, As, Bs, T(0), Cs)
end

function gemm_batched(tA::Char, tB::Char, alpha::T, A::CuArray{T,3}, B::CuArray{T,3}) where T
    m = size(A, tA == 'N' ? 1 : 2)
    n = size(B, tB == 'N' ? 2 : 1)
    C = CuArray{T}(m, n, size(A,3))
    gemm_batched!(tA, tB, alpha, A, B, T(0), C)
end
