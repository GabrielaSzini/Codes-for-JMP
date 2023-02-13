using ParallelUtilities
using Distributions
using ForwardDiff
using StatsFuns
using StatFiles
using StatsModels
using DataFrames
using DelimitedFiles
using ProgressMeter
using ThreadsX
using Base.Iterators
using Random
using LinearAlgebra
using Optim, NLSolversBase
using Base.Threads
using FileIO, JLD2
using StatsModels
using CategoricalArrays
using RDatasets
using CSV
using StatsBase
#using ModelFrame
using GLM


using Distributed
addprocs(30)

@everywhere begin
    using Pkg
    Pkg.activate(".")
    using Distributions
    using ForwardDiff
    using StatsFuns
    using StatFiles
    using DataFrames
    using DelimitedFiles
    using ProgressMeter
    using ThreadsX
    using Base.Iterators
    using Random
    using LinearAlgebra
    using Optim, NLSolversBase
    using Base.Threads
    using FileIO, JLD2
    using StatsModels
    using CategoricalArrays
    using RDatasets
    using CSV
    using StatsBase
    #using ModelFrame
    using GLM

    include("utils/stats.jl")

    ### Load data ###
    trade_data = DataFrame(load("trade.dta"))
    ### Data Generating Process ###
    # removing missing values
    trade_data_dgp = dropmissing(trade_data, "vtrade")
    # winsorizing variable vtrade
    #vtrade_dgp = trade_data_dgp[!, 16]
    #upper_threshold = quantile(vtrade_dgp, 0.95)
    #vtrade_dgp_winsorized = vtrade_dgp
    #vtrade_dgp_winsorized[vtrade_dgp_winsorized .> upper_threshold] .= upper_threshold
    #trade_data_dgp[!,:"ytrim"] = vtrade_dgp_winsorized
    # run tobit with two-way FE
    #cat_array_id = categorical(trade_data_dgp[!, :id])
    #cat_array_jd = categorical(trade_data_dgp[!, :jd])
    #trade_data_dgp[!, :id] = cat_array_id
    #trade_data_dgp[!, :jd] = cat_array_jd

    # obtaining Tobit estimates from CFW
    coefficients_tobit_label = CSV.read("coefficients_tobit_labels.csv", DataFrame)
    coefficients_tobit = coefficients_tobit_label[!, :x]
    #coefficients_tobit_label = coefficients_tobit_label[!, :Column1]
    scale_tobit = CSV.read("scale_tobit.csv", DataFrame)
    scale_tobit = scale_tobit[1,1]

    # rescaling coefficients
    sigma_tobit = scale_tobit
    theta_tobit = (coefficients_tobit./sigma_tobit).*(pi/sqrt(3))

    # constructing ys (quantiles of data)
    ys = quantile((trade_data_dgp[!, "vtrade"]), collect(0:100)/100)
    # take only from 110 onwards (last 0 quantile onwards)
    ys = ys[55:101]
    ys_tobit = ys./sigma_tobit.*(pi/sqrt(3))
    taus = collect(0:100)/100
    taus = taus[55:101]
    n_thres = length(ys)

    # construct new data
    nobs = size(trade_data_dgp)[1]
    # construct set of dummys
    #dummy_id = ModelFrame(@formula(vtrade ~ 0 + id), trade_data_dgp, contrasts = Dict(:id => DummyCoding())) |> modelmatrix
    #dummy_jd = ModelFrame(@formula(vtrade ~ 1 + jd), trade_data_dgp, contrasts = Dict(:jd => DummyCoding())) |> modelmatrix
    #dummy_jd = dummy_jd[:, 2:end]
    # construct equivalent of W matrix: to obtain Ysim
    #W_matrix = hcat(trade_data_dgp[!,"ldist"], trade_data_dgp[!,"legal"], trade_data_dgp[!,"border"], trade_data_dgp[!,"language"], trade_data_dgp[!,"colony"], trade_data_dgp[!,"currency"], trade_data_dgp[!,"fta"], trade_data_dgp[!,"religion"], dummy_id, dummy_jd)
    
    W_matrix = CSV.read("W_matrix.csv", DataFrame)
    W_matrix = Matrix(W_matrix)

    # for explanatory variables in estimation method
    trade_data_matrix = Matrix(trade_data)
    N=157
    #remaining of explanatory variables we simply get from original dataset
    # log of distance
    X₁_vector = trade_data_matrix[:,8]
    X₁ = reshape(X₁_vector, N, N)
    # colonial ties
    X₂_vector = trade_data_matrix[:,11]
    X₂ = reshape(X₂_vector, N, N)
    # currency union
    X₃_vector = trade_data_matrix[:,7]
    X₃ = reshape(X₃_vector, N, N)
    # FTA
    X₄_vector = trade_data_matrix[:,9]
    X₄ = reshape(X₄_vector, N, N)
    # border
    X₅_vector = trade_data_matrix[:,6]
    X₅ = reshape(X₅_vector, N, N)
    # legal system
    X₆_vector = trade_data_matrix[:,4]
    X₆ = reshape(X₆_vector, N, N)
    # language
    X₇_vector = trade_data_matrix[:,3]
    X₇ = reshape(X₇_vector, N, N)
    # religion
    X₈_vector = trade_data_matrix[:,5]
    X₈ = reshape(X₈_vector, N, N)

    X₁ = coalesce.(X₁, 0)
    X₂ = coalesce.(X₂, 0)
    X₃ = coalesce.(X₃, 0)
    X₄ = coalesce.(X₄, 0)
    X₅ = coalesce.(X₅, 0)
    X₆ = coalesce.(X₆, 0)
    X₇ = coalesce.(X₇, 0)
    X₈ = coalesce.(X₈, 0)

end

@everywhere function simulation(nobs, W_matrix, theta_tobit, ys_tobit, n_thres, taus, X₁, X₂, X₃, X₄, X₅, X₆, X₇, X₈, sim)
    # UP TO HERE EVERYTHING IS THE SAME NO MATTER WHICH SIMULATION
    # start generating data
    u = randn(nobs)
    norm = Normal()
    v = cdf(norm, u)
    v = inverselogit.(v)

    # generate new dependent variable
    Ysim = W_matrix*theta_tobit + v
    Ysim .= max.(Ysim, 0)

    # then now just compute as usual with new Ysim and covariates
    # first we need to put back the diagonal elements
    N = 157
    Y = fill(NaN, (N,N))
    count = 1
    for i in 1:N
        for j in 1:N
            if i != j
                # Replace the elements in Y[i,j] with the corresponding elements in Y_vector[count]
                Y[i,j] = Ysim[count]
                count += 1
            end
        end
    end

    # after obtaining the grids we can replace missing values with zeros as they do not matter (only diagonal)
    Y = coalesce.(Y, 0)
    Y = replace_NaN(Y)

    # make dummy considering ys_tobit as grid
    Ydummy_matrix = dummy(Y, N, ys_tobit, n_thres)

    #storing results for one SIMULATION
    β̂_comb_grid  = Array{Float64}(undef, n_thres, 8)
    β̂_comb_grid .= NaN
    se₁_grid = Array{Float64}(undef, n_thres, 8)
    se₁_grid .= NaN
    quantile_grid = Array{Float64}(undef, n_thres, 1)
    quantile_grid .= NaN
    percentage_nodes = Array{Float64}(undef, n_thres, 1)
    percentage_nodes .= NaN
    simulation_number = Array{Float64}(undef, n_thres, 1)
    simulation_number .= sim

    # running through quantiles
    for g in 1:n_thres
        quantile_grid[g,1] = taus[g]
        Y_1 = Ydummy_matrix[:,:,g]
        percentage_nodes[g,1] = (sum(Y_1) - 157)/(157*156)

        index_i_comb, index_j_comb, ỹ_comb, b_comb, c_comb, nondiag_comb, X̃₁_comb, X̃₂_comb, X̃₃_comb, X̃₄_comb, X̃₅_comb, X̃₆_comb, X̃₇_comb, X̃₈_comb =  dataquantilecomb(Y_1, X₁, X₂, X₃, X₄, X₅, X₆, X₇, X₈)

        X̃₁_comb_cond = @view X̃₁_comb[c_comb.==1]
        X̃₂_comb_cond = @view X̃₂_comb[c_comb.==1]
        X̃₃_comb_cond = @view X̃₃_comb[c_comb.==1]
        X̃₄_comb_cond = @view X̃₄_comb[c_comb.==1]
        X̃₅_comb_cond = @view X̃₅_comb[c_comb.==1]
        X̃₆_comb_cond = @view X̃₆_comb[c_comb.==1]
        X̃₇_comb_cond = @view X̃₇_comb[c_comb.==1]
        X̃₈_comb_cond = @view X̃₈_comb[c_comb.==1]
        ỹ_comb_cond = @view ỹ_comb[c_comb.==1]
        nondiag_comb_cond = @view nondiag_comb[c_comb.==1]

        X̃_comb_cond = [X̃₁_comb_cond X̃₂_comb_cond X̃₃_comb_cond X̃₄_comb_cond X̃₅_comb_cond X̃₆_comb_cond X̃₇_comb_cond X̃₈_comb_cond]

        β₀_comb = zeros(8,1)
        β₀_comb = initialize!(β₀_comb, X̃_comb_cond, ỹ_comb_cond)
        nll_comb = make_closures(X̃_comb_cond, ỹ_comb_cond)
        result_comb = optimize(nll_comb, β₀_comb, LBFGS(), autodiff=:forward)
        β̂_comb = Optim.minimizer(result_comb)

        β̂_comb_grid[g,:] = β̂_comb'

        se₁ = standarderrors_application(Y_1, X₁, X₂, X₃, X₄, X₅, X₆, X₇, X₈, β̂_comb, X̃_comb_cond,nondiag_comb_cond)
        se₁_grid[g,:] = se₁'
    end

    output = hcat(β̂_comb_grid, se₁_grid, quantile_grid, percentage_nodes, ys_tobit, simulation_number)

    file_path = "results_simulations/simulation_output_$(sim).jld2"
    save(file_path, "output", output)

    return output

end

sims = 100
simulation_output_total = @time @showprogress pmap(51:sims) do sim
    simulation(nobs, W_matrix, theta_tobit, ys_tobit, n_thres, taus, X₁, X₂, X₃, X₄, X₅, X₆, X₇, X₈, sim)
end

save("results_simulations/simulation_output_total.jld2","simulation_output_total",simulation_output_total)



#check if passing everything needed inside function