"""
	Linear operator for block matrix B = [M  N; 0  M]
	Applies B to vector v = [x; y] → [M*x + N*y; M*y]
"""
struct BlockOperatorFrechet{TM,TN}
    M::TM
    N::TN
    n::Int
end

function LinearAlgebra.mul!(out, B::BlockOperatorFrechet, v)
    n = B.n
    x = @view v[1:n]
    y = @view v[(n+1):2n]

    outx = @view out[1:n]
    outy = @view out[(n+1):2n]

    # outx = M*x
    mul!(outx, B.M, x)

    # outx += N*y
    mul!(outx, B.N, y, 1.0, 1.0)

    # outy = M*y
    mul!(outy, B.M, y)

    return out
end

Base.size(B::BlockOperatorFrechet) = (2*B.n, 2*B.n)
Base.size(B::BlockOperatorFrechet, d::Int) = d <= 2 ? 2*B.n : 1  # Add this!
Base.eltype(B::BlockOperatorFrechet{TM,TN}) where {TM,TN} =
    promote_type(eltype(TM), eltype(TN))

# Tell Julia this is not Hermitian
LinearAlgebra.ishermitian(B::BlockOperatorFrechet) = false
LinearAlgebra.issymmetric(B::BlockOperatorFrechet) = false

"""
Compute Fréchet derivative L(A, E) via block exponential. (Al Mohy and Higham)
"""
function frechet(A::Matrix, E::Matrix)
    n = size(A, 1)
    # Block matrix: [A E; 0 A]
    M = [A E; zeros(n, n) A]
    expM = exp(M)
    # Extract blocks
    expA = expM[1:n, 1:n]
    L = expM[1:n, (n+1):end]
    return expA, L
end


"""
	Compute Fréchet sensitivity: sens = L_exp(M, N) * p
	using block expv. Only returns sensitivity (not prediction).
	"""
function frechet_sens_expv(M, N, p; tol = 1e-10, m = 30)
    n = length(p)
    v = zeros(eltype(p), 2n)
    @view(v[(n+1):2n]) .= p

    B = BlockOperatorFrechet(M, N, n)
    w = expv(1.0, B, v; tol = tol, m = m)

    return @view(w[1:n])  # Sensitivity only
end
