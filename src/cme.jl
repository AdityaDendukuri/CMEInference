
# =============================================================================
# BLOCK 2: Build Generator (Simple - Direct Parameterization)
# =============================================================================

begin
    """
    Build generator with DIRECT rate parameterization.
    θ[k] = rate for stoichiometry k (must be ≥ 0)
    A[i,j] = θ[k] where k is the stoichiometry class of (i→j)
    """
    function build_generator_simple(θ, states, stoich_list, stoich_to_trans)
        n = length(states)
        A = zeros(n, n)

        # Fill off-diagonal entries
        for (k, Δ) in enumerate(stoich_list)
            # Rate for this stoichiometry (DIRECT, not log-space)
            rate = θ[k]

            # Apply to all transitions with this stoichiometry
            for (i, j) in stoich_to_trans[Δ]
                A[i, j] = rate
            end
        end

        # Fill diagonal (column sums = 0)
        for j = 1:n
            A[j, j] = -sum(A[i, j] for i = 1:n if i != j)
        end

        return A
    end

    println("Generator construction (simple - direct rates) defined")
end
