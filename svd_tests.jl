"
Author: Daniel O'Connor
Email: dt.oconnor@outlook.com

File re-creating previous SVD work and new work.
"
include("samplers/tools.jl")
include("samplers/SA.jl")
include("samplers/SVMC.jl")
include("samplers/SVMC_TF.jl")
include("samplers/pimc.jl")
include("dynamic_models/spin_vector_O2_dynamics.jl")
include("dynamic_models/spin_vector_O3_dynamics.jl")

using Random
using ..SA
using ..SVMC
using ..SVMCTF
using ..PIMC
using ..SamplingTools
using ..SpinVectorO2Dynamics
using ..SpinVectorO3Dynamics
using LinearAlgebra
using DifferentialEquations
using Profile
using BenchmarkTools
using NPZ
using Interpolations
using Plots
using LsqFit


function closed_O2()
    """
    simulating code from here
    https://arxiv.org/pdf/1305.4904.pdf
    """
    h = Dict()
    j = Dict()
    for i =1:8
        if i <= 4
            h[i] = -1.0
            j[(i, i+4)] = -1.0
            j[(i, (i%4) + 1)] = -1.0
        else
            h[i] = 1.0
        end
    end

    s = collect(0:0.001:1)
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

    run_time = 1000.0
    anneal_time = 1000.0    # for T = 1000 in paper
    result, sol = SpinVectorO2Dynamics.solve(h, j, anneal_time=anneal_time,
                                         run_time=run_time, a_fn=a_fn,
                                         b_fn=b_fn, saveat=s)

    run_time = 1000.0
    anneal_time = 200.0     # for T = 200 in paper
    result2, sol2 = SpinVectorO2Dynamics.solve(h, j, anneal_time=anneal_time,
                                      run_time=run_time, a_fn=a_fn,
                                      b_fn=b_fn, saveat=s)

    final_state, energy = result
    final_state2, energy2 = result2
    # ts is the times at which are saved on the ODE solver
    ts = run_time .* s
    thetas_in = zeros(Float64, length(ts))
    thetas_out = zeros(Float64, length(ts))
    thetas_in2 = zeros(Float64, length(ts))
    thetas_out2 = zeros(Float64, length(ts))
    for i = eachindex(ts)
        thetas_in[i] = mean(sol.u[i].x[1][1:4]) / pi
        thetas_out[i] = mean(sol.u[i].x[1][5:8]) / pi
        thetas_in2[i] = mean(sol2.u[i].x[1][1:4]) / pi
        thetas_out2[i] = mean(sol2.u[i].x[1][5:8]) / pi
    end

    print("T = 200 ns, solution = $(final_state2), energy = $(energy2)\n",
          "T = 1000 ns, solution = $(final_state), energy = $(energy) ")

    plot(ts, thetas_in, xaxis="t / ns", yaxis="θ / π", legend=false, ylims=[-0.1, 1.1], linecolor=:red)
    plot!(ts, thetas_out, linecolor=:red, legend=false)
    plot!(ts, thetas_in2, linecolor=:black, legend=false)
    plot!(ts, thetas_out2, linecolor=:black, legend=false)
end


function simulate_O3()
    """
    simulating code from here (plots are inverted due to the different
    signs in the Hamiltonian)
    https://arxiv.org/pdf/1403.4228.pdf
    """
    h = Dict()
    j = Dict()
    nspins = 8
    for i =1:8
        if i <= 4
            h[i] = -1.0
            j[(i, i+4)] = -1.0
            j[(i, (i%4) + 1)] = -1.0
        else
            h[i] = 1.0
        end
    end

    temp = 2.226 / (2*pi)

    s = collect(0:0.0001:1)
    num_steps = length(s)
    # approximate the schedule
    #  0.1099 * b(s=1) = temp = 17 mK
    #  0.2834 * b(s≈0.541) = temp
    a0 = 34.25 / (2*pi)
    b0 = 1 / (2*pi)
    bt = temp / 0.1099
    at = 1e-9   # assumption
    # form of b = quadratic
    b_fn = (x) -> (2.28 * x^2) + (0.78 * x) +0.01 # + (1/(2*pi))
    sa = [0.0, 0.38, 0.541, 0.8, 0.9, 1]
    a = [a0, 5/(2*pi), temp, 3e-4, 2e-6, 1e-9]
    a_fn = SamplingTools.monotonic_interpolation(sa, a)
    # a_fn = (x) -> 0.024074802432616997 * exp(5.427221233043444*(1.0 - x).^2) -
    #         0.024074802432616997

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

    # max_field = a0
    # a_sch= max_field .* (1.0 .- s)
    # b_sch= max_field .* s
    # a_fn = SamplingTools.monotonic_interpolation(s, a_sch)
    # b_fn = SamplingTools.monotonic_interpolation(s, b_sch)
    #
    run_time = 20000.0
    anneal_time = 20000.0
    # open system enables multi-threading for parallel computation
    num_samples = 10
    open_sys = false
    friction_constant = 1e-6
    bloch = true
    if open_sys
        atol, reltol = 1e-2, 1e-2
    else
        atol, reltol = 1e-9, 1e-6
    end

    # Long anneal times require very low tolerances
    results, sol = SpinVectorO3Dynamics.solve(h, j, anneal_time=anneal_time,
                                             run_time=run_time, a_fn=a_fn,
                                             b_fn=b_fn, temp=temp,
                                             friction_constant=friction_constant,
                                             trajectories=num_samples,
                                             saveat=s,
                                             open_system=open_sys, Bloch=bloch,
                                             abstol=atol, reltol=reltol,
                                             initial_theta_state=fill(0.5 * pi, 8),
                                             initial_phi_state=zeros(Float64, 8))

    # ts is the times at which are saved on the ODE solver
    ts = LinRange(0.0, run_time, num_steps)
    # plot(sol)
    if open_sys
        final_states, energies, occurences = results
        print("Solutions = $(final_states) \n energies = $(energies) \n occurences = $(occurences)")
        med = DifferentialEquations.EnsembleAnalysis.timeseries_point_median(sol, ts)
        # low = DifferentialEquations.EnsembleAnalysis.timeseries_point_quantile(sol, 0.025, ts)
        # high = DifferentialEquations.EnsembleAnalysis.timeseries_point_quantile(sol, 0.975, ts)
        print(size(med), size(s))
        magnetization = med[16:24, :]'
    else
        final_state, energy = results
        magnetization = zeros(Float64, num_steps, length(h))
        if !bloch
            print("Solution = $(final_state) \n energy = $(energy) \n final θs = $(sol.u[end].x[1])")
            for i = 1:num_steps
                magnetization[i, :] = cos.(sol.u[i].x[1])
            end
        else
            print("Solution = $(final_state) \n energy = $(energy) \n final ⟨σ⟩ = $(sol.u[end][(2*nspins + 1):(3*nspins)])")
            for i = 1:num_steps
                magnetization[i, :] = sol.u[i][(2*nspins + 1):(3*nspins)]
            end
        end
    end
    plot(s, magnetization, xaxis="\$ t  / t_f \$", yaxis="\$ \\langle \\sigma ^z \\rangle  \$", legend=false)
end


function main_dynamics()
    d = 0.1
    N = 2
    h, j = SamplingTools.PFC_dict(d, N=N)
    num_samples = 1000
    temp = 0.01226 * 1.380649e-23 / (1e9 * 6.62607015e-34)

    s = collect(0:0.001:1)
    num_steps = length(s)
    max_field = 3.0
    a_sch= max_field .* (1.0 .- s)
    b_sch= max_field .* s
    a_fn = SamplingTools.monotonic_interpolation(s, a_sch)
    b_fn = SamplingTools.monotonic_interpolation(s, b_sch)

    schedule_data = npzread("data\\DWAVE_schedule.npz")
    a_fn = SamplingTools.monotonic_interpolation(schedule_data[:, 1], 0.5.*schedule_data[:, 2])
    b_fn = SamplingTools.monotonic_interpolation(schedule_data[:, 1], 0.5.*schedule_data[:, 3])
    a_sch = a_fn.(s)
    b_sch = b_fn.(s)

    # # DQA schedule
    # a_sch, b_sch = SamplingTools.DQA_schedule(1, 2*N, a_sch, b_sch,
    #                                           sx=0.2, cx=0.0, c1=0.0)
    # # convert to functions
    # a_fn = Dict([(i,  SamplingTools.monotonic_interpolation(s, a_sch[i])) for i=1:nspins])
    # b_fn = Dict([(i,  SamplingTools.monotonic_interpolation(s, b_sch[i])) for i=1:nspins])

    run_time = 20000.0
    anneal_time = 20000.0
    # open system enables multi-threading for parallel computation
    open_sys = false
    friction_constant = 2e-2

    # anneal_time = 1000.0

    results, sol = SpinVectorO2Dynamics.solve(h, j, anneal_time=anneal_time,
                                             run_time=run_time, a_fn=a_fn,
                                             b_fn=b_fn, temp=temp, saveat=s,
                                             friction_constant=friction_constant,
                                             trajectories=num_samples,
                                             open_system=open_sys,
                                             abstol=1e-7, reltol=1e-6)

    # ts is the times at which are saved on the ODE solver
    ts = LinRange(0.0, run_time, num_steps)
    if open_sys
        final_states, energies, occurences = results
        print("Solutions = $(final_states) \n energies = $(energies) \n occurences = $(occurences)")
        med = DifferentialEquations.EnsembleAnalysis.timeseries_point_median(sol, ts)
        # low = DifferentialEquations.EnsembleAnalysis.timeseries_point_quantile(sol, 0.025, ts)
        # high = DifferentialEquations.EnsembleAnalysis.timeseries_point_quantile(sol, 0.975, ts)
        magnetization = cos.(med[1:(2*N), :]')

    else
        final_state, energy = results
        print("Solution = $(final_state) \n energy = $(energy) \n final θs = $(sol.u[end].x[1])")
        magnetization = zeros(Float64, num_steps, length(h))
        for i = 1:num_steps
            magnetization[i, :] = cos.(sol.u[i].x[1])
        end
    end
    plot(ts, magnetization, xaxis="\$ t \\, / \\,  ns \$", yaxis="\$ \\langle \\sigma ^z \\rangle  \$", legend=false, ylims=[-1, 1])
end
