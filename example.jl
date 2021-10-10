"
Author: Daniel O'Connor
Email: dt.oconnor@outlook.com

File of how to use the algorithms currently implemented.
"
include("samplers/tools.jl")
include("samplers/SA.jl")
include("samplers/SVMC.jl")
include("samplers/SVMC_TF.jl")
include("samplers/pimc.jl")
include("dynamic_models/spin_vector_O2_dynamics.jl")

using Random
using ..SA
using ..SVMC
using ..SVMCTF
using ..PIMC
using ..SamplingTools
using ..SpinVectorO2Dynamics
using LinearAlgebra
using DifferentialEquations
using Profile
using BenchmarkTools
using NPZ
using Interpolations
using Plots


function main()
    # test 0 =  SA, test 1 = SVMC, test 2 = SVMCTF, test 3 = PIMC
    test = 3

    # simple PFC problem
    d = 0.1
    N = 2
    h, J = SamplingTools.PFC_dict(d, N=N)

    initial_state = Array{Float64, 1}([-1.0 for i = 1:(2*N)])

    num_samples = 1000
    num_sweeps = 1000
    num_slices = 20
    temp = 0.01226 * 1.380649e-23 / (1e9 * 6.62607015e-34)

    a_sch= 1.0 .- collect(LinRange(0, 1, num_sweeps))
    b_sch= collect(LinRange(0, 1, num_sweeps))

    # schedule_data = npzread("data\\DWAVE_schedule.npz")
    # a_fn = SamplingTools.monotonic_interpolation(schedule_data[:, 1], 0.5.*schedule_data[:, 2])
    # b_fn = SamplingTools.monotonic_interpolation(schedule_data[:, 1], 0.5.*schedule_data[:, 3])
    # a_sch = a_fn.(collect(LinRange(0, 1, num_sweeps)))
    # b_sch = b_fn.(collect(LinRange(0, 1, num_sweeps)))

    # DQA schedule
    a_sch, b_sch = SamplingTools.DQA_schedule(1, 2*N, a_sch, b_sch,
                                              sx=0.2, cx=0.0, c1=0.0)

    if test == 0
        # SIMULATED ANNEALING TEST
        # result = SA.sample(h, J, num_reads=num_samples, num_steps=num_sweeps,
                           # add_xtalk=false, noisy=false)
        result = SA.sample(h, J, num_reads=num_samples, num_steps=num_sweeps,
                           add_xtalk=false, noisy=true,
                           # geometric schedule
                           temp_schedule=exp.(LinRange(log(10), log(0.001), num_sweeps)),
                           initial_state=initial_state)
    elseif test == 1
        # SVMC TEST
        # result = SVMC.sample(h, J, num_reads=num_samples, temp=temp,
        #                      a_sch=a_sch, b_sch=b_sch)
        result = SVMC.sample(h, J, num_reads=num_samples, temp=temp,
                             a_sch=a_sch, b_sch=b_sch, add_xtalk=false,
                             noisy=false, spherical=true,)
                             # initial_state=initial_state)
    elseif test == 2
        # SVMC-TF TEST
        # result = SVMCTF.sample(h, J, num_reads=num_samples, temp=temp,
        #                      a_sch=a_sch, b_sch=b_sch)
        result = SVMCTF.sample(h, J, num_reads=num_samples, temp=temp,
                             a_sch=a_sch, b_sch=b_sch, add_xtalk=false,
                             noisy=false, spherical=true,
                             initial_state=initial_state)
    elseif test == 3
        # if initial state given then it superseeds the pre-anneal
        result = PIMC.sample(h, J, num_reads=num_samples, temp=temp,
                             a_sch=a_sch, b_sch=b_sch, nslices=num_slices;
                             add_xtalk=false, noisy=false, pre_anneal=false,)
                             # initial_state=initial_state)
    end
    print("Transverse field supression factor = $(supression_coeff)\n")
    return  result
end


function main_dynamics()
    d = 0.01
    N = 2
    nspins = 2*N
    h, j = SamplingTools.PFC_dict(d, N=N)
    num_samples = 1000
    temp = 0.01226 * 1.380649e-23 / (1e9 * 6.62607015e-34)
    s = collect(0:0.001:1)
    num_steps = length(s)

    max_field = 1.0
    a_sch= max_field .* (1.0 .- s)
    b_sch= max_field .* s
    a_fn = SamplingTools.monotonic_interpolation(s, a_sch)
    b_fn = SamplingTools.monotonic_interpolation(s, b_sch)

    # schedule_data = npzread("data\\DWAVE_schedule.npz")
    # a_fn = SamplingTools.monotonic_interpolation(schedule_data[:, 1], 0.5.*schedule_data[:, 2])
    # b_fn = SamplingTools.monotonic_interpolation(schedule_data[:, 1], 0.5.*schedule_data[:, 3])
    # a_sch = a_fn.(s)
    # b_sch = b_fn.(s)

    # # DQA schedule
    # a_sch, b_sch = SamplingTools.DQA_schedule(1, 2*N, a_sch, b_sch,
    #                                           sx=0.2, cx=0.0, c1=0.0)
    # # convert to functions
    # a_fn = Dict([(i,  SamplingTools.monotonic_interpolation(s, a_sch[i])) for i=1:nspins])
    # b_fn = Dict([(i,  SamplingTools.monotonic_interpolation(s, b_sch[i])) for i=1:nspins])
    #
    run_time = 1000.0
    anneal_time = 1000.0
    # open system enables multi-threading for parallel computation
    open_sys = true
    friction_constant = 2e-2

    # anneal_time = 1000.0
    results, sol = SpinVectorO2Dynamics.solve(h, j, anneal_time=anneal_time,
                                         run_time=run_time, a_fn=a_fn,
                                         b_fn=b_fn, temp=temp,
                                         friction_constant=friction_constant,
                                         trajectories=num_samples,
                                         open_system=open_sys, abstol=1e-8,
                                         reltol=1e-6)

    # ts is the times at which are saved on the ODE solver
    ts = LinRange(0.0, run_time, num_steps)
    if open_sys
        final_states, energies, occurences = results
        print("Solutions = $(final_states) \n energies = $(energies) \n occurences = $(occurences)")
        med = DifferentialEquations.EnsembleAnalysis.timeseries_point_median(sol, ts)
        # low = DifferentialEquations.EnsembleAnalysis.timeseries_point_quantile(sol, 0.025, ts)
        # high = DifferentialEquations.EnsembleAnalysis.timeseries_point_quantile(sol, 0.975, ts)
        thetas = med[1:(2*N), :]' ./pi
        omegas = med[(2*N + 1):(4*N), :]'
    else
        final_state, energy = results
        print("Solution = $(final_state) \n energy = $(energy) \n final θs = $(sol.u[end].x[1])")
        thetas = zeros(Float64, num_steps, length(h))
        omegas = zeros(Float64, num_steps, length(h))
        for i = 1:num_steps
            thetas[i, :] = sol.u[i].x[1] ./ pi
            omegas[i, :] = sol.u[i].x[2]
        end
    end
    plt1 = plot(ts, thetas, xaxis="t / ns", yaxis="θ / π", legend=false)
    plt2 = plot(ts, omegas, xaxis="t / ns", yaxis="ω", legend=false)
    plot(plt1, plt2, layout=(2, 1))
end
