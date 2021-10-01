"
Author: Daniel O'Connor
Email: dt.oconnor@outlook.com

"


module SpinVectorO2Dynamics
    include("..\\samplers\\tools.jl")

    using ..SamplingTools: hj_info, add_noise, add_cross_talk, rotor_aggregation
    using DifferentialEquations
    using Random
    using LinearAlgebra
    using Interpolations
    using RecursiveArrayTools


    function O2_model(spin_vector::Array{Float64, 1},
                             biases::Array{Float64, 1},
                             JM::Array{Float64, 2},
                             run_time::Float64,
                             independent_schedules::Bool,
                             open_system::Bool, # stochastic only
                             temp::Float64, # stochastic only
                             damping::Float64, # stochastic only
                             trajectories::Int64, # stochastic only
                             at_fn,
                             bt_fn,
                             ts)
        nspins = size(spin_vector, 1)
        ω_0 = zeros(Float64, nspins)
        u_0 = ArrayPartition(spin_vector, ω_0)
        t_span = (0.0, run_time)

        model! = function (du, u, p, t)
            # θ'(t) = ω(t)
            du.x[1] .= u.x[2]
            # α/I = ω'(t) = -1/I dV/dϑ = F/I
            du.x[2] .= (at_fn(t) .* cos.(u.x[1])) +
                       (bt_fn(t) .* ((biases .* sin.(u.x[1])) +
                                     (sin.(u.x[1]) .* (JM * cos.(u.x[1])))))
            nothing
        end

        model_jac! = function (J, u, p, t)
            # Jacobian J[i, j] = df_i / du_j
            J[1, 1, :] .= 0.0
            J[1, 2, :] .= 1.0
            J[2, 2, :] .= 0.0
            J[2, 1, :] .= (-at_fn(t) .* sin.(u.x[1])) +
                         (-bt_fn(t) .* ((biases .* cos.(u.x[1])) +
                                         (cos.(u.x[1]) .* (JM * cos.(u.x[1])))))
            nothing
        end

        noise_model! = function (du, u, p, t)
            # θ'(t) = ω(t)
            du.x[1] .= u.x[2]
            # ω'(t) = - dV/dϑ - λω = F/I
            du.x[2] .= (-damping .* u.x[2]) +
                       (at_fn(t) .* cos.(u.x[1])) +
                       (bt_fn(t) .* ((biases .* sin.(u.x[1])) +
                                     (sin.(u.x[1]) .* (JM * cos.(u.x[1])))))
            nothing
        end

        noise_model_jac! = function (J, u, p, t)
            # Jacobian J[i, j] = df_i / du_j
            J[1, 1, :] .= 0.0
            J[1, 2, :] .= 1.0
            J[2, 2, :] .= -damping
            J[2, 1, :] .= (-at_fn(t) .* sin.(u.x[1])) +
                         (-bt_fn(t) .* ((biases .* cos.(u.x[1])) +
                                         (cos.(u.x[1]) .* (JM * cos.(u.x[1])))))
            nothing
        end

        σ_model! = function (du, u, p, t)
            # constants infront of wiener process
            du.x[1] .= 0.0
            du.x[2] .= sqrt(2*damping*temp)
            nothing
        end

        # supplying jacobian can provide speed increases, especially for stiff
        # problems, use commented line to remove jacobian version

        if open_system
            f =  DifferentialEquations.ODEFunction(noise_model!, jac=noise_model_jac!)
            # f = noise_model!
            prob = DifferentialEquations.SDEProblem(f, σ_model!, u_0, t_span)
            ensembleprob = EnsembleProblem(prob)
            # can be upgraded to use GPU
            sol = DifferentialEquations.solve(ensembleprob, EnsembleThreads(),
                                              trajectories=trajectories,
                                              saveat=ts)
        else
            f =  DifferentialEquations.ODEFunction(model!, jac=model_jac!)
            # f = model!
            prob = DifferentialEquations.ODEProblem(f, u_0, t_span)
            sol = DifferentialEquations.solve(prob, saveat=ts)
        end

        return sol
    end


    function solve(h::Dict, J::Dict; anneal_time=10.0, run_time=0.0, # in nanoseconds
                    temp=0.1, friction_constant=1e-3, trajectories=1000,
                    # for individual qubit schedules, input a
                    # dictionary of schedules
                    a_sch=1.0 .- collect(LinRange(0, 1, 100)),
                    b_sch=collect(LinRange(0, 1, 100)),
                    add_xtalk=false, noisy=false, open_system=false,
                    initial_state=Array{Float64, 1}())

        nspins = length(h)
        # check to see if we have individual qubit schedules
        independent_qubit_sechdules = false
        if isa(a_sch, Dict) | isa(b_sch, Dict)
            # if one input is not a dict but one if, convert to dict of lists
            if !isa(a_sch, Dict)
                schedule = deepcopy(a_sch)
                a_sch = Dict([(i, schedule) for i = 1:nspins])
            elseif !isa(b_sch, Dict)
                schedule = deepcopy(b_sch)
                b_sch = Dict([(i, schedule) for i = 1:nspins])
            end

            # check consistent inputs
            @assert length(a_sch) == nspins && length(b_sch) == nspins
            num_steps = length(a_sch[1])
            @assert all([length(a_sch[i]) == num_steps for i = 1:nspins])
            @assert all([length(b_sch[i]) == num_steps for i = 1:nspins])

            independent_qubit_sechdules = true
        else
            @assert length(a_sch) == length(b_sch)
            num_steps = length(a_sch)
        end

        # assign the time domain, can either be a simple number or array
        @assert isa(anneal_time, Int) || isa(anneal_time, Float64) || isa(anneal_time, Array)
        if isa(anneal_time, Array)
            @assert all(sort(anneal_time) .== anneal_time)
            @assert length(anneal_time) == length(num_steps)
        else
            anneal_time = LinRange(0.0, anneal_time, num_steps)
        end
        t_anneal = anneal_time[end]
        if run_time == 0
            run_time = t_anneal
        end

        # define coefficient interpolation functions
        if independent_qubit_sechdules
            at_sch_all = Dict()
            bt_sch_all = Dict()
            for i = 1:nspins
                at_sch_all[i] = LinearInterpolation(anneal_time, a_sch[i], extrapolation_bc=Flat())
                bt_sch_all[i] = LinearInterpolation(anneal_time, b_sch[i], extrapolation_bc=Flat())
            end
            # vectorise the independent_schedules
            at_fn = function (t)
                a_vec = zeros(Float64, nspins)
                for i = 1:nspins
                    a_vec[i] = at_sch_all[i](t)
                end
                return a_vec
            end
            bt_fn = function (t)
                b_vec = zeros(Float64, nspins)
                for i = 1:nspins
                    b_vec[i] = bt_sch_all[i](t)
                end
                return b_vec
            end
        else
            at_fn = LinearInterpolation(anneal_time, a_sch, extrapolation_bc=Flat())
            bt_fn = LinearInterpolation(anneal_time, b_sch, extrapolation_bc=Flat())
        end

        # prepare initial state if defined
        if length(initial_state) == nspins
            @assert ndims(initial_state) == 1
            spin_vector = acos.(deepcopy(initial_state))
        else
            # starting in the transverse field direction as default
            spin_vector = fill(0.5 * pi, nspins)
        end

        # add noise effects
        if noisy
            h, J = add_noise(h, J)
        end
        # add cross-talk effects
        if add_xtalk
            h, J = add_xtalk(h, J)
        end

        # prepare problem info
        JM, M, degs = hj_info(h, J)
        JS = deepcopy(JM)
        biases = diag(JM)
        JM -= diagm(biases)
        JM += JM'
        save_times = LinRange(0.0, run_time, num_steps)

        # run the protocol
        sol = O2_model(spin_vector, biases, JM, run_time,
                      independent_qubit_sechdules,
                      open_system, # stochastic only
                      temp, # stochastic only
                      friction_constant, # stochastic only
                      trajectories, # stochastic only
                      at_fn, bt_fn, save_times)

        if open_system
            spin_vectors = [sol[i].u[end].x[1] for i =1:trajectories]
            results = rotor_aggregation(spin_vectors, trajectories, collect(UpperTriangular(JS)))
        else
            final_sol = sol.u[end].x[1]
            spins = Array{Float64, 1}(sign.(cos.(final_sol)))
            energy = biases' * spins + 0.5 * (spins' * (JM * spins))
            results = [spins, energy]
        end
        return results, sol
    end

    export solve, O2_model

end
