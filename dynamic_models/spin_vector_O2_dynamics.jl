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
                             open_system::Bool, # stochastic only
                             temp::Float64, # stochastic only
                             damping::Float64, # stochastic only
                             trajectories::Int64, # stochastic only
                             at_fn,
                             bt_fn,
                             ts,
                             abstol::Float64,
                             reltol::Float64)
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
            # f =  DifferentialEquations.ODEFunction(model!, jac=model_jac!)
            f = model!
            prob = DifferentialEquations.ODEProblem(f, u_0, t_span)
            sol = DifferentialEquations.solve(prob, saveat=ts, abstol=abstol,
                                              reltol=reltol)
        end

        return sol
    end


    function solve(h::Dict, J::Dict; anneal_time=100.0, run_time=0.0, # in nanoseconds
                    temp=0.1, friction_constant=1e-3, trajectories=1000,
                    # for individual qubit schedules, input a
                    # dictionary of interpolations
                    # NOTE if interpolations aren't continuous then
                    # instabilities arise in the ODE integrator
                    a_fn=nothing, b_fn=nothing, saveat=collect(0:0.01:1),
                    add_xtalk=false, noisy=false, open_system=false,
                    initial_state=Array{Float64, 1}(), abstol=1e-8,
                    reltol=1e-6)

        nspins = length(h)
        # check to see if we have individual qubit schedules
        independent_qubit_sechdules = false
        if isa(a_fn, Dict) | isa(b_fn, Dict)
            # if one input is not a dict but one if, convert to dict of lists
            if !isa(a_fn, Dict)
                interp = deepcopy(a_fn)
                a_fn = Dict([(i, interp) for i = 1:nspins])
            elseif !isa(b_fn, Dict)
                interp = deepcopy(b_fn)
                b_fn = Dict([(i, interp) for i = 1:nspins])
            end

            independent_qubit_sechdules = true
        elseif a_fn == nothing
            # linear schedules
            a_fn = LinearInterpolation(saveat, 1.0 .- saveat, extrapolation_bc=Flat())
        elseif b_fn == nothing
            b_fn = LinearInterpolation(saveat, saveat, extrapolation_bc=Flat())
        end

        if run_time == 0
            run_time = anneal_time
        end

        # define coefficient interpolation functions
        if independent_qubit_sechdules
            # vectorise the independent_schedules
            at_fn = function (t)
                a_vec = zeros(Float64, nspins)
                for i = 1:nspins
                    a_vec[i] = a_fn[i](t/anneal_time)
                end
                return a_vec
            end
            bt_fn = function (t)
                b_vec = zeros(Float64, nspins)
                for i = 1:nspins
                    b_vec[i] = b_fn[i](t/anneal_time)
                end
                return b_vec
            end
        else
            at_fn = (t) -> a_fn(t/anneal_time)
            bt_fn = (t) -> b_fn(t/anneal_time)
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

        # run the protocol
        sol = O2_model(spin_vector, biases, JM, run_time,
                      open_system, # stochastic only
                      temp, # stochastic only
                      friction_constant, # stochastic only
                      trajectories, # stochastic only
                      at_fn, bt_fn, run_time.*saveat, abstol, reltol)

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
