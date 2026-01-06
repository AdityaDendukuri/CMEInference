
"""
Compute empirical probability distributions from trajectory snapshots.
"""
function compute_distributions_from_trajectories(trajectories, states, snapshot_times)
    state_to_idx = Dict(s => i for (i, s) in enumerate(states))
    n_states = length(states)
    n_snapshots = length(snapshot_times)

    distributions = [zeros(n_states) for _ = 1:n_snapshots]

    for traj in trajectories
        for (snap_idx, t_snap) in enumerate(snapshot_times)
            # Find state at time t_snap
            state_at_t = traj.u[1]  # Default to initial

            for (i, t) in enumerate(traj.t)
                if t <= t_snap
                    state_at_t = traj.u[i]
                else
                    break
                end
            end

            # Convert to CartesianIndex
            if isa(state_at_t, Vector) || isa(state_at_t, Tuple)
                state_at_t = CartesianIndex(state_at_t...)
            end

            # Add to distribution
            idx = get(state_to_idx, state_at_t, nothing)
            if idx !== nothing
                distributions[snap_idx][idx] += 1.0
            end
        end
    end

    # Normalize
    for dist in distributions
        total = sum(dist)
        if total > 0
            dist ./= total
        end
    end

    return distributions
end


"""
Perturbation E_k = ∂A/∂θ_k for stoichiometry class k.
Since A[i,j] = θ_k directly, we have ∂A[i,j]/∂θ_k = 1
"""
function build_perturbation_simple(k, θ, states, stoich_list, stoich_to_trans)
    n = length(states)
    E = zeros(n, n)

    Δ = stoich_list[k]

    # ∂A[i,j]/∂θ_k = 1 for all (i,j) with stoichiometry Δ
    for (i, j) in stoich_to_trans[Δ]
        E[i, j] = 1.0
        E[j, j] -= 1.0  # Diagonal adjustment
    end

    return E
end

"""
Extract all states visited during a time window.
"""
function extract_states_in_window(trajectories, t_start, t_end)
    states_set = Set{CartesianIndex{4}}()
    state_counts = Dict{CartesianIndex{4},Int}()

    for traj in trajectories
        for (idx, t) in enumerate(traj.t)
            if t_start <= t <= t_end
                state = traj.u[idx]

                # Convert to CartesianIndex
                if isa(state, Vector) || isa(state, Tuple)
                    state = CartesianIndex(state...)
                end

                push!(states_set, state)
                state_counts[state] = get(state_counts, state, 0) + 1
            end
        end
    end

    # Sort by frequency
    sorted_states = sort(collect(states_set), by = s->get(state_counts, s, 0), rev = true)

    return sorted_states, state_counts
end



"""
Extract transitions observed during time window.
"""
function extract_transitions_in_window(trajectories, states, t_start, t_end)
    state_to_idx = Dict(s => i for (i, s) in enumerate(states))
    transition_counts = Dict{Tuple{Int,Int},Int}()

    for traj in trajectories
        for idx = 1:(length(traj.t)-1)
            t = traj.t[idx]

            # Check if transition occurs in window
            if t_start <= t < t_end
                state_from = traj.u[idx]
                state_to = traj.u[idx+1]

                # Convert
                if isa(state_from, Vector) || isa(state_from, Tuple)
                    state_from = CartesianIndex(state_from...)
                end
                if isa(state_to, Vector) || isa(state_to, Tuple)
                    state_to = CartesianIndex(state_to...)
                end

                # Map to indices (note: A[i,j] means j→i)
                i = get(state_to_idx, state_to, nothing)
                j = get(state_to_idx, state_from, nothing)

                if i !== nothing && j !== nothing && i != j
                    transition_counts[(i, j)] = get(transition_counts, (i, j), 0) + 1
                end
            end
        end
    end

    return transition_counts
end


"""
	Build empirical distributions at snapshot times within window.
"""
function build_snapshots_in_window(trajectories, states, t_start, t_end, dt)
    state_to_idx = Dict(s => i for (i, s) in enumerate(states))
    n_states = length(states)

    # Snapshot times
    snapshot_times = collect(t_start:dt:t_end)
    n_snapshots = length(snapshot_times)

    # Distributions
    distributions = [zeros(n_states) for _ = 1:n_snapshots]

    for traj in trajectories
        for (snap_idx, t_snap) in enumerate(snapshot_times)
            # Find state at this time
            state_at_t = traj.u[1]

            for (i, t) in enumerate(traj.t)
                if t <= t_snap
                    state_at_t = traj.u[i]
                else
                    break
                end
            end

            # Convert
            if isa(state_at_t, Vector) || isa(state_at_t, Tuple)
                state_at_t = CartesianIndex(state_at_t...)
            end

            # Add to distribution
            idx = get(state_to_idx, state_at_t, nothing)
            if idx !== nothing
                distributions[snap_idx][idx] += 1.0
            end
        end
    end

    # Normalize
    for dist in distributions
        total = sum(dist)
        if total > 0
            dist ./= total
        end
    end

    return snapshot_times, distributions
end

function group_by_stoichiometry(transition_counts, states)
    stoich_to_transitions = Dict{Vector{Int},Vector{Tuple{Int,Int}}}()
    transition_to_stoich = Dict{Tuple{Int,Int},Vector{Int}}()

    for (i, j) in keys(transition_counts)
        # Compute stoichiometry: Δ = state_i - state_j (i is target, j is source)
        Δ = collect(Tuple(states[i])) - collect(Tuple(states[j]))

        # Skip null transitions
        if all(Δ .== 0)
            continue
        end

        # Group by stoichiometry
        if !haskey(stoich_to_transitions, Δ)
            stoich_to_transitions[Δ] = []
        end
        push!(stoich_to_transitions[Δ], (i, j))

        transition_to_stoich[(i, j)] = Δ
    end

    return stoich_to_transitions, transition_to_stoich
end
