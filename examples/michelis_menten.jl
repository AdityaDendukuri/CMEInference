begin
    # numerical libraries
    using ExponentialUtilities, SparseArrays
    # output and plotting
    using ProgressLogging, JLD, CairoMakie
    # modelling and statistics 
    using Catalyst, JumpProcesses, StatsBase
    using Interpolations
    # importing local fsp package
    using Revise, Optim
    using ExponentialUtilities
    include("../src/DiscStochInf.jl")
end

# Reaction network
rn = @reaction_network begin
    kB, S + E --> SE
    kD, SE --> S + E
    kP, SE --> P + E
end

generate_ssa_data = begin
    local u0_integers = [:S => 50, :E => 10, :SE => 1, :P => 1]
    local tspan = (0.0, 50.0)
    local ps = [:kB => 0.01, :kD => 0.1, :kP => 0.1]
    global rates_true = [0.01, 0.1, 0.1]  # For comparison later

    jinput = JumpInputs(rn, u0_integers, tspan, ps)
    jprob = JumpProblem(jinput)

    n_trajs = 10000
    trajectories = []
    @progress for i = 1:n_trajs
        push!(trajectories, solve(jprob, SSAStepper()))
    end

    println("Generated $(length(trajectories)) SSA trajectories")
end

begin
    println("\n" * "="^70)
    println("EXTRACTING WINDOW 1 DATA")
    println("="^70)

    snapshot_dt = 1.0      # Snapshot spacing within window
    n_windows = 1          # Total number of windows

    t_start = 0.0
    t_end = 1.0

    # Extract state space
    window_states, _ = extract_states_in_window(trajectories, t_start, t_end)

    # Extract transitions
    window_transitions =
        extract_transitions_in_window(trajectories, window_states, t_start, t_end)

    # Group by stoichiometry
    window_stoich_to_trans, _ = group_by_stoichiometry(window_transitions, window_states)

    window_stoich_list = sort(
        collect(keys(window_stoich_to_trans)),
        by = s->length(window_stoich_to_trans[s]),
        rev = true,
    )

    # Build snapshots
    snapshot_times, snapshot_dists =
        build_snapshots_in_window(trajectories, 
                                  window_states,
                                  t_start,
                                  t_end, snapshot_dt)

    println("Window 1: [$(t_start) → $(t_end)]")
    println("  States: $(length(window_states))")
    println("  Stoichiometries: $(length(window_stoich_list))")
    for (k, Δ) in enumerate(window_stoich_list)
        n_trans = length(window_stoich_to_trans[Δ])
        println("    Jump $k: Δ=$Δ ($n_trans transitions)")
    end
    println("  Snapshots: $(length(snapshot_dists))")

    # Choose basis degree
    basis_degree = 2  # Start with linear: [1, S, E, SE, P]

    # Count features
    test_state = Tuple(window_states[1])
    n_features = length(evaluate_polynomial_basis(test_state, basis_degree))
    n_params_total = length(window_stoich_list) * n_features

    println("\n" * "="^60)
    println("BASIS FUNCTIONS")
    println("="^60)
    println("Degree: $basis_degree")
    println("Features per jump: $n_features")
    println("Jump directions: $(length(window_stoich_list))")
    println("Total parameters: $n_params_total")

    # Example
    println("\nExample basis at state $(collect(test_state)):")
    println("  $(evaluate_polynomial_basis(test_state, basis_degree))")
end

begin
    println("\n" * "="^70)
    println("EMPIRICAL INITIALIZATION")
    println("="^70)

    # Compute state occupancy
    state_occupancy = Dict{CartesianIndex{4},Float64}()

    for traj in trajectories
        for idx = 1:(length(traj.t)-1)
            t = traj.t[idx]
            if t_start <= t < t_end
                state = traj.u[idx]
                if isa(state, Vector) || isa(state, Tuple)
                    state = CartesianIndex(state...)
                end
                dt_traj = traj.t[idx+1] - traj.t[idx]
                state_occupancy[state] = get(state_occupancy, state, 0.0) + dt_traj
            end
        end
    end

    θ0 = zeros(length(window_stoich_list) * n_features)

    for (k, Δ) in enumerate(window_stoich_list)
        transitions = window_stoich_to_trans[Δ]

        total_rate = 0.0
        n_valid = 0

        for (i, j) in transitions
            state_j = window_states[j]
            count_ij = get(window_transitions, (i, j), 0)
            time_j = get(state_occupancy, state_j, 0.0)

            if time_j > 0 && count_ij > 0
                rate_ij = count_ij / time_j
                S, E, SE, P = Tuple(state_j)

                if Δ == [-1, -1, 1, 0] && S > 0 && E > 0
                    total_rate += rate_ij / (S * E)
                    n_valid += 1
                elseif (Δ == [1, 1, -1, 0] || Δ == [0, 1, -1, 1]) && SE > 0
                    total_rate += rate_ij / SE
                    n_valid += 1
                end
            end
        end

        emp_rate = n_valid > 0 ? total_rate / n_valid : 0.1

        # Set appropriate basis coefficient
        if Δ == [-1, -1, 1, 0]
            θ0[(k-1)*n_features+7] = emp_rate  # S×E
        else
            θ0[(k-1)*n_features+4] = emp_rate  # SE
        end

        println("  Jump $k (Δ=$Δ): empirical = $(round(emp_rate, digits=6))")
    end

    println("\nInitialized $(length(θ0)) parameters")
end

begin
    println("\n" * "="^70)
    println("OPTIMIZATION: Exact Fréchet Gradient")
    println("="^70)

    function objective(θ)
        obj, _ = objective_gradient_frechet_exact(
            θ,
            window_states,
            window_stoich_list,
            window_stoich_to_trans,
            snapshot_dists,
            snapshot_times,
            basis_degree,
            n_features;
            use_l2 = false,
        )
        return obj
    end

    function gradient!(G, θ)
        _, grad = objective_gradient_frechet_exact(
            θ,
            window_states,
            window_stoich_list,
            window_stoich_to_trans,
            snapshot_dists,
            snapshot_times,
            basis_degree,
            n_features;
            use_l2 = false,
        )
        G .= grad
    end

    result_frechet = optimize(
        objective,
        gradient!,
        θ0,
        LBFGS(),
        Optim.Options(iterations = 5000, show_every = 50, show_trace = true, g_tol = 1e-7),
    )

    θ_learned = Optim.minimizer(result_frechet)

    println("\n" * "="^70)
    println("RESULTS")
    println("="^70)
    println("Converged: $(Optim.converged(result_frechet))")
    println("Objective: $(round(Optim.minimum(result_frechet), digits=6))")
    println("Iterations: $(Optim.iterations(result_frechet))")

    println("\nLearned parameters:")
    for (k, Δ) in enumerate(window_stoich_list)
        θ_k = θ_learned[((k-1)*n_features+1):(k*n_features)]

        if Δ == [-1, -1, 1, 0]
            println("  Jump $k (S×E): $(round(θ_k[7], digits=6)) (true: 0.01)")
        elseif Δ == [1, 1, -1, 0]
            println("  Jump $k (SE):  $(round(θ_k[4], digits=6)) (true: 0.1)")
        elseif Δ == [0, 1, -1, 1]
            println("  Jump $k (SE):  $(round(θ_k[4], digits=6)) (true: 0.1)")
        end
    end
end
