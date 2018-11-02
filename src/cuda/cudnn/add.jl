"""
C = α*A + β*C

The bias tensor A must match the corresponding dimension of the destination tensor
C or must be equal to 1.
"""
function add!(α, A::CuArray{T}, β, C::CuArray{T}) where T
    h = gethandle()
    adesc = TensorDesc(A, 4)
    cdesc = TensorDesc(C, 4)
    @cudnn(:cudnnAddTensor,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr),
        h, T[α], adesc, A, T[β], cdesc, C)
    C
end
