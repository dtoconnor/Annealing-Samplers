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
using Random
using ..SA
using ..SVMC
using ..SVMCTF
using ..PIMC
using ..SamplingTools
using LinearAlgebra
using Profile
using BenchmarkTools
using NPZ
using Interpolations


function main()
    # test 0 =  SA, test 1 = SVMC, test 2 = SVMCTF, test 3 = PIMC
    test = 2

    # simple PFC problem
    d = 0.1
    h = Dict([(1, -1.0), (2, 1.0 - d), (3, 1.0-d), (4, -1.0)])
    J = Dict([((1, 2), -1.0), ((2, 3), -1.0), ((3, 4), -1.0)])

    initial_state = Array{Float64, 1}([1.0, -1.0, -1.0, 1.0])

    num_samples = 1000
    num_sweeps = 1000
    num_slices = 20
    temp = 0.01226 * 1.380649e-23 / (1e9 * 6.62607015e-34)

    # a_sch= 1.0 .- collect(LinRange(0, 1, num_sweeps))
    # b_sch= collect(LinRange(0, 1, num_sweeps))

    schedule_data = npzread("data\\DWAVE_schedule.npz")
    a_fn = LinearInterpolation(schedule_data[:, 1], 0.5.*schedule_data[:, 2])
    b_fn = LinearInterpolation(schedule_data[:, 1], 0.5.*schedule_data[:, 3])
    a_sch = a_fn.(collect(LinRange(0, 1, num_sweeps)))
    b_sch = b_fn.(collect(LinRange(0, 1, num_sweeps)))

    # inidividual qubit schedules
    # a_sch = Dict([(1, deepcopy(a_sch)), (2, 1e-6 .* deepcopy(a_sch)),
    #                    (3, deepcopy(a_sch)), (4, deepcopy(a_sch)),])
    # b_sch = Dict([(1, deepcopy(b_sch)), (2, deepcopy(b_sch)),
    #                    (3, deepcopy(b_sch)), (4, deepcopy(b_sch)),])

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

    return  result
end
