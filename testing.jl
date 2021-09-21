"
Author: Daniel O'Connor
Email: dt.oconnor@outlook.com

File used to profile and benchmark the algorithms implmented.
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
    test = 3

    # simple PFC problem
    d = 0.1
    h = Dict([(1, -1.0), (2, 1.0 - d), (3, 1.0-d), (4, -1.0)])
    J = Dict([((1, 2), -1.0), ((2, 3), -1.0), ((3, 4), -1.0)])

    num_samples = 1000
    num_sweeps = 10000
    num_slices = 20
    nspins = length(h)
    temp = 0.01226 * 1.380649e-23 / (1e9 * 6.62607015e-34)

    a_sch= 1.0 .- collect(LinRange(0, 1, num_sweeps))
    b_sch= collect(LinRange(0, 1, num_sweeps))

    # schedule_data = npzread("data\\DWAVE_schedule.npz")
    # a_fn = LinearInterpolation(schedule_data[:, 1], 0.5.*schedule_data[:, 2])
    # b_fn = LinearInterpolation(schedule_data[:, 1], 0.5.*schedule_data[:, 3])
    # a_sch = a_fn.(collect(LinRange(0, 1, num_sweeps)))
    # b_sch = b_fn.(collect(LinRange(0, 1, num_sweeps)))

    a_sch_dict = Dict([(1, deepcopy(a_sch)), (2, 0.0 .* deepcopy(a_sch)),
                       (3, deepcopy(a_sch)), (4, deepcopy(a_sch)),])
    b_sch_dict = Dict([(1, deepcopy(b_sch)), (2, deepcopy(b_sch)),
                       (3, deepcopy(b_sch)), (4, deepcopy(b_sch)),])

    # initialize the problem parameter
    JM, M, degs = SamplingTools.hj_info(h, J)
    nbs_idxs, nbs_vals = SamplingTools.generate_nbr(nspins, M, maximum(degs) + 1)

    hot_t, cold_t = SamplingTools.default_temp_range(JM)
    temp_schedule = collect(LinRange(hot_t, cold_t, num_sweeps))

    # initialize state
    initial_state = Array{Float64, 1}([1.0, -1.0, -1.0, -1.0])
    if test == 0
        # spin_vectors = [deepcopy(initial_state) for i =1:num_samples]
        initial_state = rand([-1.0, 1.0], nspins, num_samples)
        spin_vectors = [initial_state[:, i] for i =1:num_samples]
        _fn = SA.anneal!
        args_in = [nbs_idxs, nbs_vals, temp_schedule]
    elseif test == 1 || test == 2
        # spin_vectors = [acos.(deepcopy(initial_state)) for i =1:num_reads]
        spin_vectors = fill(0.5 * pi, nspins, num_samples)
        spin_vectors = [spin_vectors[:, i] for i =1:num_samples]
        if test == 1
            _fn = SVMC.anneal!
        else
            _fn = SVMCTF.anneal!
        end
        args_in = [nbs_idxs, nbs_vals, a_sch, b_sch, temp]
    elseif  test == 3
        # spin_array = repeat(initial_state', num_slices)'
        # spin_vectors = [copy(spin_array) for i =1:num_samples]
        initial_state = rand([-1.0, 1.0], nspins, num_slices, num_samples)
        spin_vectors = [initial_state[:, :, i] for i =1:num_samples]
        _fn = PIMC.anneal!
        args_in = [nbs_idxs, nbs_vals, a_sch, b_sch, temp]
    end

    function run()
        @inbounds for i = 1:num_samples
            _fn(spin_vectors[i], args_in...)
        end
    end

    run()

    if test == 0
        print(discrete_aggregation(spin_vectors, num_samples, collect(UpperTriangular(JM))))
    elseif test == 1 || test == 2
        print(rotor_aggregation(spin_vectors, num_samples, collect(UpperTriangular(JM))))
    elseif  test == 3
        print(pimc_aggregation(spin_vectors, num_samples, collect(UpperTriangular(JM))))
    end

    p = @profiler run()
    Profile.print()

    return  @benchmark $run()
end
