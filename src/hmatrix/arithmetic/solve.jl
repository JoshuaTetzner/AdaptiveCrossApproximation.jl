"""
    SolveWorkspace(F::HMatrixLU)
    SolveWorkspace(workspace::SolveWorkspace)

Buffers for solves with the factorization `F`. It acts like `F`:
`ldiv!(S, b)`, `ldiv!(x, S, b)`, `S \\ b`, `ldiv!(transpose(S), b)` and
`ldiv!(S', b)`, for vectors and matrices of right-hand sides. Solves with `F`
itself build a temporary workspace on every call. One workspace serves one
solve at a time; `SolveWorkspace(workspace)` gives another task its own.
"""
struct SolveWorkspace{T,F<:HMatrixLU{T}} <: LinearAlgebra.Factorization{T}
    factor::F
    scratch::BlockScratch{T}
end

SolveWorkspace(F::HMatrixLU{T}) where {T} = SolveWorkspace(F, BlockScratch{T}())
SolveWorkspace(S::SolveWorkspace) = SolveWorkspace(S.factor)

Base.size(S::SolveWorkspace) = size(S.factor)
Base.size(S::SolveWorkspace, dim::Integer) = size(S.factor, dim)
LinearAlgebra.issuccess(S::SolveWorkspace) = issuccess(S.factor)

"""
    _lusolve!(F::HMatrixLU, B::StridedMatrix, scratch, transposed)

Solve `A*X = B` or, with `transposed`, `transpose(A)*X = B` in place for `B` in
external order. `B` is permuted to tree order, where `A = P'*L*U` and
`transpose(A) = transpose(U)*transpose(L)*P`, solved with the dense triangular
solves `_trisolve!`, and permuted back.
"""
function _lusolve!(F::HMatrixLU{T}, B::StridedMatrix{T}, scratch, transposed) where {T}
    issuccess(F) || throw(ArgumentError("hierarchical LU factorization failed"))
    size(B, 1) == size(F, 1) || throw(DimensionMismatch("right-hand side size differs"))
    _applytreeswaps!(B, F.treeswaps, false)
    try
        if transposed
            _trisolve!(B, F.upper, scratch, Val(:T), Val(:left))
            _trisolve!(B, F.lower, scratch, Val(:T), Val(:left))
            _pivot!(B, F.lower; inverse=true)
        else
            _pivot!(B, F.lower)
            _trisolve!(B, F.lower, scratch, Val(:N), Val(:left))
            _trisolve!(B, F.upper, scratch, Val(:N), Val(:left))
        end
    finally
        _applytreeswaps!(B, F.treeswaps, true)
    end
    return B
end

_solvematrix(B::StridedMatrix) = B
_solvematrix(B::StridedVector) = reshape(B, :, 1)

_solveparts(S::SolveWorkspace) = (S.factor, S.scratch)
_solveparts(F::HMatrixLU{T}) where {T} = (F, BlockScratch{T}())

const _LUSolver{T} = Union{HMatrixLU{T},SolveWorkspace{T}}

function LinearAlgebra.ldiv!(S::_LUSolver{T}, B::StridedVecOrMat{T}) where {T}
    F, scratch = _solveparts(S)
    _lusolve!(F, _solvematrix(B), scratch, false)
    return B
end

function LinearAlgebra.ldiv!(
    S::LinearAlgebra.TransposeFactorization{T,<:_LUSolver{T}}, B::StridedVecOrMat{T}
) where {T}
    F, scratch = _solveparts(Base.parent(S))
    _lusolve!(F, _solvematrix(B), scratch, true)
    return B
end

# A^H x = b is solved as A^T conj(x) = conj(b); for real matrices A^H = A^T.
function LinearAlgebra.ldiv!(
    S::LinearAlgebra.AdjointFactorization{T,<:_LUSolver{T}}, B::StridedVecOrMat{T}
) where {T}
    F, scratch = _solveparts(Base.parent(S))
    T <: Real || conj!(B)
    _lusolve!(F, _solvematrix(B), scratch, true)
    T <: Real || conj!(B)
    return B
end

"""
    H \\ B

Solve `H*X = B` with a hierarchical LU factorization of a copy of `H`; `H` is
unchanged. Each call factorizes again: for several solves with the same matrix,
keep `F = lu(H)` and solve with `F \\ B`.
"""
Base.:\(H::HMatrixCore, B::AbstractVecOrMat) = lu(H) \ B
