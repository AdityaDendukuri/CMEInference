
"""
	Compute gradient using adjoint method.
	
	Mathematical derivation:
	  obj = Σₖ |p_{k+1}^data - exp(A*dt) p_k|₁
	  
	  Let F(θ) = exp(A(θ)*dt) p_k
	  ∂obj/∂θᵢ = -sign(residual)' * ∂F/∂θᵢ
	  
	  Using adjoint method:
	  ∂F/∂θᵢ = exp(A*dt) * (∂A/∂θᵢ) * p_k
	  
	  Define adjoint: λ such that exp(A*dt)' λ = -sign(residual)
	  Then: ∂obj/∂θᵢ = λ' * (∂A/∂θᵢ) * p_k
	"""

function objective_gradient_adjoint(θ)
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

    n = length(window_states)
    n_params = length(θ)

    obj = 0.0
    grad = zeros(n_params)

    dt_train = train_times_fsp[2] - train_times_fsp[1]

    # Compute propagator once
    P = exp(Matrix(A * dt_train))
    P_transpose = transpose(P)

    # Loop over windows
    for win_idx = 1:(length(train_dists_fsp)-1)
        # Forward propagation
        p_curr = train_dists_fsp[win_idx]
        p_pred = P * p_curr
        residual = train_dists_fsp[win_idx+1] - p_pred

        # Objective
        obj += sum(abs.(residual))

        # Adjoint solve: P' λ = -sign(residual)
        rhs = -sign.(residual)
        λ = P_transpose \ rhs  # Solve adjoint equation

        # Gradient via adjoint
        for k = 1:length(window_stoich_list)
            for f = 1:n_features
                param_idx = (k-1)*n_features + f

                # Compute ∂A/∂θᵢ (perturbation matrix)
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

                # Gradient: grad[i] = λ' * E * p_curr * dt
                # (Note: need dt because ∂exp(A*dt)/∂A involves dt)
                grad[param_idx] += dot(λ, E * (p_curr * dt_train))
            end
        end
    end

    # Regularization
    λ_reg = 1e-4
    obj += λ_reg * sum(abs2, A)

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
            grad[param_idx] += 2 * λ_reg * sum(A .* E)
        end
    end

    return obj, grad
end



"""
	Compute objective and gradient using EXACT Fréchet derivatives.
	
	Uses block expv to compute L_exp(M, N_i) * p for each parameter.
	This is EXACT (no adjoint approximation).
	"""
function objective_gradient_frechet_exact(
    θ,
    window_states,
    window_stoich_list,
    window_stoich_to_trans,
    snapshot_dists,
    snapshot_times,
    basis_degree,
    n_features;
    krylov_tol = 1e-10,
    krylov_m = 30,
    use_l2 = false,
    smooth_l1_eps = 1e-8,
)

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

    n = size(A, 1)
    obj = 0.0
    grad = zeros(length(θ))

    dt_snap = snapshot_times[2] - snapshot_times[1]
    M = dt_snap * A  # Keep sparse!

    # Loop over snapshots
    for snap_idx = 1:(length(snapshot_dists)-1)
        p_curr = snapshot_dists[snap_idx]
        p_next = snapshot_dists[snap_idx+1]

        # Forward prediction (once per snapshot)
        pred = expv(1.0, M, p_curr; tol = krylov_tol, m = krylov_m)
        residual = p_next - pred

        # Objective
        if use_l2
            obj += 0.5 * dot(residual, residual)
            u = -residual
        else
            # Smooth L1
            smooth_abs = sqrt.(residual .^ 2 .+ smooth_l1_eps^2)
            obj += sum(smooth_abs)
            u = -(residual ./ smooth_abs)
        end

        # Gradient: loop over parameters
        for k = 1:length(window_stoich_list)
            for f = 1:n_features
                param_idx = (k-1)*n_features + f

                # Build perturbation (sparse)
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
                N = dt_snap * E

                # Exact Fréchet action via Krylov
                sens = frechet_sens_expv(M, N, p_curr; tol = krylov_tol, m = krylov_m)

                # Gradient: u' * sens
                grad[param_idx] += dot(u, sens)
            end
        end
    end

    # Regularization
    λ_reg = 1.0e-7
    obj += λ_reg * sum(abs, A)

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
            grad[param_idx] += 2 * λ_reg * sum(A .* E)
        end
    end

    return obj, grad
end



function objective_gradient_simple(
    θ,
    windows,
    states,
    stoich_list,
    stoich_to_trans,
    dt;
    λ_frob = 1e-4,
)
    A = build_generator_simple(θ, states, stoich_list, stoich_to_trans)
    n = size(A, 1)
    n_params = length(stoich_list)

    # Check validity
    if any(!isfinite, A)
        return Inf, zeros(n_params)
    end

    obj = 0.0
    grad = zeros(n_params)

    # Compute propagator
    local P
    try
        P = exp(Matrix(A * dt))
    catch
        return Inf, zeros(n_params)
    end

    # Prediction error
    for window in windows
        p_pred = P * window.p_curr
        residual = window.p_next - p_pred
        obj += sum(abs.(residual))

        # Gradient via Fréchet
        for k = 1:n_params
            E = build_perturbation_simple(k, θ, states, stoich_list, stoich_to_trans)
            _, L = frechet(A * dt, E * dt)
            grad[k] -= dot(sign.(residual), L * window.p_curr)
        end
    end

    # Frobenius regularization
    if λ_frob > 0
        obj += λ_frob * sum(abs2, A)
        for k = 1:n_params
            E = build_perturbation_simple(k, θ, states, stoich_list, stoich_to_trans)
            grad[k] += 2 * λ_frob * sum(A .* E)
        end
    end

    return obj, grad
end











"""
  Verify gradient against finite differences.
"""
function gradient_check_frechet(
    θ,
    window_states,
    window_stoich_list,
    window_stoich_to_trans,
    snapshot_dists,
    snapshot_times,
    basis_degree,
    n_features;
    ε = 1e-7,
    n_check = 5,
)

    println("\n" * "="^70)
    println("GRADIENT CHECK: Exact Fréchet vs Finite Difference")
    println("="^70)

    # Exact gradient
    obj0, grad_exact = objective_gradient_frechet_exact(
        θ,
        window_states,
        window_stoich_list,
        window_stoich_to_trans,
        snapshot_dists,
        snapshot_times,
        basis_degree,
        n_features;
        use_l2 = true,
    )

    println("\nObjective: $(round(obj0, digits=6))")
    println("Gradient norm: $(round(norm(grad_exact), digits=6))")

    # Check random parameters
    n_params = length(θ)
    indices = rand(1:n_params, min(n_check, n_params))

    println("\nChecking $(length(indices)) parameters:")
    println("  Param | Exact | FD | Rel Error")
    println("  " * "-"^50)

    max_error = 0.0

    for i in indices
        # Forward
        θ_plus = copy(θ)
        θ_plus[i] += ε
        obj_plus, _ = objective_gradient_frechet_exact(
            θ_plus,
            window_states,
            window_stoich_list,
            window_stoich_to_trans,
            snapshot_dists,
            snapshot_times,
            basis_degree,
            n_features;
            use_l2 = true,
        )

        # Backward
        θ_minus = copy(θ)
        θ_minus[i] -= ε
        obj_minus, _ = objective_gradient_frechet_exact(
            θ_minus,
            window_states,
            window_stoich_list,
            window_stoich_to_trans,
            snapshot_dists,
            snapshot_times,
            basis_degree,
            n_features;
            use_l2 = true,
        )

        grad_fd = (obj_plus - obj_minus) / (2ε)
        grad_ex = grad_exact[i]

        rel_error = abs(grad_fd - grad_ex) / (abs(grad_ex) + 1e-10)
        max_error = max(max_error, rel_error)

        println(
            "  $(rpad(i, 5)) | $(rpad(round(grad_ex, digits=4), 8)) | " *
            "$(rpad(round(grad_fd, digits=4), 8)) | $(round(rel_error*100, digits=2))%",
        )
    end

    println("\nMax error: $(round(max_error*100, digits=3))%")
    println(max_error < 1e-3 ? "✓ PASSED" : "✗ FAILED")

    return max_error
end
