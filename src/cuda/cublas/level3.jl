import Base.LinAlg.BLAS: gemm, gemm!

for (fname, elty) in ((:cublasDgemm,:Float64), (:cublasSgemm,:Float32))
    @eval begin
        function gemm!(tA::Char, tB::Char,
            alpha::$elty, A::CuVecOrMat{$elty}, B::CuVecOrMat{$elty},
            beta::$elty, C::CuVecOrMat{$elty})

            @assert device(A) == device(B) == device(C)
            m = size(A, tA == 'N' ? 1 : 2)
            k = size(A, tA == 'N' ? 2 : 1)
            n = size(B, tB == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2) || k != size(B, tB == 'N' ? 1 : 2)
                throw(DimensionMismatch())
            end
            $fname(handle(C), cublasop(tA), cublasop(tB), m, n, k,
                $elty[alpha], A, stride(A,2), B, stride(B,2), $elty[beta], C, stride(C,2))
            C
        end
    end
end
function gemm{T}(tA::Char, tB::Char, alpha::T, A::CuVecOrMat{T}, B::CuVecOrMat{T})
    C = similar(B, size(A, tA=='N' ? 1 : 2), size(B, tB=='N' ? 2 : 1))
    gemm!(tA, tB, alpha, A, B, T(0), C)
end
#gemm{T}(tA::Char, tB::Char, A::CuVecOrMat{T}, B::CuVecOrMat{T}) = gemm(tA, tB, T(1), A, B)
#gemm{T}(A::CuVecOrMat{T}, B::CuVecOrMat{T}; tA='N', tB='N', alpha=1) = gemm(tA, tB, T(alpha), A, B)

for (fname,elty) in ((:cublasDgemmBatched,:Float64), (:cublasSgemmBatched,:Float32))
    @eval begin
        function gemm_batched!(tA::Char, tB::Char,
            alpha::$elty, As::Vector{CuMatrix{$elty}}, Bs::Vector{CuMatrix{$elty}},
            beta::$elty, Cs::Vector{CuMatrix{$elty}})

            if (length(As) != length(Bs) || length(As) != length(Cs))
                throw(DimensionMismatch(""))
            end
            for i = 1:length(As)
                A, B, C = As[i], Bs[i], Cs[i]
                m = size(A, tA == 'N' ? 1 : 2)
                k = size(A, tA == 'N' ? 2 : 1)
                n = size(B, tB == 'N' ? 2 : 1)
                if m != size(C,1) || n != size(C,2) || k != size(B, tB == 'N' ? 1 : 2)
                    throw(DimensionMismatch(""))
                end
            end
            m = size(As[1], tA == 'N' ? 1 : 2)
            k = size(As[1], tA == 'N' ? 2 : 1)
            n = size(Bs[1], tB == 'N' ? 2 : 1)
            lda = max(1, stride(As[1],2))
            ldb = max(1, stride(Bs[1],2))
            ldc = max(1, stride(Cs[1],2))
            Aptrs = map(a -> Ptr{$elty}(a.ptr), As)
            Bptrs = map(a -> Ptr{$elty}(a.ptr), Bs)
            Cptrs = map(a -> Ptr{$elty}(a.ptr), Cs)
            $fname(handle(C), cublasop(tA), cublasop(tB), m, n, k, [alpha], pointer(Aptrs),
                lda, pointer(Bptrs), ldb, [beta], pointer(Cptrs), ldc, length(As))
            Cs
        end
    end
end
function gemm_batched{T}(tA::Char, tB::Char,
    alpha::T, A::Vector{CuVecOrMat{T}}, B::Vector{CuVecOrMat{T}})
    C = CudaMatrix{T}[similar(B[1], (size(A[1], tA=='N' ? 1 : 2), size(B[1], tB=='N' ? 2 : 1))) for i in 1:length(A)]
    gemm_batched!(tA, tB, alpha, A, B, T(0), C)
end
function gemm_batched{T}(tA::Char, tB::Char,
    A::Vector{CuVecOrMat{T}}, B::Vector{CuVecOrMat{T}})
    gemm_batched(tA, tB, T(1), A, B)
end
