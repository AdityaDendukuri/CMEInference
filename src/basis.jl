
"""
	Basis function types for propensities.
"""
function evaluate_polynomial_basis(state_tuple, degree)
    if degree == 0
        # Constant only
        return [1.0]
    elseif degree == 1
        # Linear: [1, S, E, SE, P]
        return [1.0; Float64.(collect(state_tuple))]
    elseif degree == 2
        # Quadratic: [1, S, E, SE, P, S², SE, E², ...]
        features = [1.0]
        x = Float64.(collect(state_tuple))

        # Linear terms
        append!(features, x)

        # Quadratic terms (upper triangle)
        for i = 1:length(x)
            for j = i:length(x)
                push!(features, x[i] * x[j])
            end
        end

        return features
    else
        error("Degree $degree not supported")
    end
end


"""
Use softplus(x) = log(1 + exp(x)) to guarantee positive propensities.
This is smooth (unlike max) and always positive.
"""
softplus(x) = log1p(exp(x))  # Numerically stable version

function build_generator_with_basis_softplus(
    θ,
    window_states,
    window_stoich_list,
    window_stoich_to_trans,
    basis_degree,
    n_features,
)
    n = length(window_states)
    A = zeros(n, n)

    for (k, Δ) in enumerate(window_stoich_list)
        θ_k = θ[((k-1)*n_features+1):(k*n_features)]

        for (i, j) in window_stoich_to_trans[Δ]
            state_j = window_states[j]
            features = evaluate_polynomial_basis(Tuple(state_j), basis_degree)

            # Use softplus instead of max
            propensity = softplus(dot(θ_k, features))

            A[i, j] = propensity
        end
    end

    for j = 1:n
        A[j, j] = -sum(A[i, j] for i = 1:n if i != j)
    end

    return A
end

function build_perturbation_with_basis_softplus(
    k,
    f,
    window_states,
    window_stoich_list,
    window_stoich_to_trans,
    basis_degree,
    n_features,
    θ,
)
    n = length(window_states)
    E = zeros(n, n)
    Δ = window_stoich_list[k]
    θ_k = θ[((k-1)*n_features+1):(k*n_features)]

    for (i, j) in window_stoich_to_trans[Δ]
        state_j = window_states[j]
        features = evaluate_polynomial_basis(Tuple(state_j), basis_degree)

        # Derivative of softplus: σ'(x) = sigmoid(x) = 1/(1+exp(-x))
        x = dot(θ_k, features)
        sigmoid_x = 1.0 / (1.0 + exp(-x))

        deriv = sigmoid_x * features[f]

        E[i, j] = deriv
        E[j, j] -= deriv
    end

    return E
end


"""
Build perturbation E_{k,f} = ∂A/∂θ_{k,f}.
This is the derivative w.r.t. the f-th basis coefficient of jump k.
"""
function build_perturbation_with_basis(
    k,
    f,
    window_states,
    window_stoich_list,
    window_stoich_to_trans,
    basis_degree,
    n_features,
    θ,
)
    n = length(window_states)
    E = zeros(n, n)
    Δ = window_stoich_list[k]

    # Get parameters for jump k
    θ_k = θ[((k-1)*n_features+1):(k*n_features)]

    for (i, j) in window_stoich_to_trans[Δ]
        state_j = window_states[j]
        features = evaluate_polynomial_basis(Tuple(state_j), basis_degree)

        # Check if propensity is positive (handles max(0, ...))
        propensity = dot(θ_k, features)

        if propensity > 0  # Only contribute if propensity is positive
            # ∂λ/∂θ_{k,f} = features[f]
            deriv = features[f]

            E[i, j] = deriv
            E[j, j] -= deriv  # Diagonal adjustment
        end
        # If propensity ≤ 0, derivative is 0 (due to max)
    end

    return E
end

function objective_gradient_basis_softplus(θ)
    A = build_generator_with_basis_softplus(
        θ,
        window_states,
        window_stoich_list,
        window_stoich_to_trans,
        basis_degree,
        n_features,
    )

    if any(!isfinite, A)
        return Inf, zeros(length(θ))
    end

    obj = 0.0
    grad = zeros(length(θ))

    dt_train = train_times_fsp[2] - train_times_fsp[1]

    for win_idx = 1:(length(train_dists_fsp)-1)
        P = exp(Matrix(A * dt_train))
        p_pred = P * train_dists_fsp[win_idx]
        residual = train_dists_fsp[win_idx+1] - p_pred

        obj += sum(abs.(residual))

        for k = 1:length(window_stoich_list)
            for f = 1:n_features
                param_idx = (k-1)*n_features + f
                E = build_perturbation_with_basis_softplus(
                    k,
                    f,
                    window_states,
                    window_stoich_list,
                    window_stoich_to_trans,
                    basis_degree,
                    n_features,
                    θ,
                )
                _, L = frechet(A * dt_train, E * dt_train)
                grad[param_idx] -= dot(sign.(residual), L * train_dists_fsp[win_idx])
            end
        end
    end

    obj += 1e-4 * sum(abs2, A)
    for k = 1:length(window_stoich_list)
        for f = 1:n_features
            param_idx = (k-1)*n_features + f
            E = build_perturbation_with_basis_softplus(
                k,
                f,
                window_states,
                window_stoich_list,
                window_stoich_to_trans,
                basis_degree,
                n_features,
                θ,
            )
            grad[param_idx] += 2 * 1e-4 * sum(A .* E)
        end
    end

    return obj, grad
end
