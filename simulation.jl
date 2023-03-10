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
using IterTools
using Random
using LinearAlgebra
using Optim, NLSolversBase
using FileIO, JLD2
using StatsModels
using CategoricalArrays
using RDatasets
using CSV
using StatsBase
#using ModelFrame
using GLM


using Distributed
addprocs(2)

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
    #using ThreadsX
    #using IterTools
    using Random
    using LinearAlgebra
    using Optim, NLSolversBase
    using FileIO, JLD2
    #using StatsModels
    #using CategoricalArrays
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
    X???_vector = trade_data_matrix[:,8]
    X??? = reshape(X???_vector, N, N)
    # colonial ties
    X???_vector = trade_data_matrix[:,11]
    X??? = reshape(X???_vector, N, N)
    # currency union
    X???_vector = trade_data_matrix[:,7]
    X??? = reshape(X???_vector, N, N)
    # FTA
    X???_vector = trade_data_matrix[:,9]
    X??? = reshape(X???_vector, N, N)
    # border
    X???_vector = trade_data_matrix[:,6]
    X??? = reshape(X???_vector, N, N)
    # legal system
    X???_vector = trade_data_matrix[:,4]
    X??? = reshape(X???_vector, N, N)
    # language
    X???_vector = trade_data_matrix[:,3]
    X??? = reshape(X???_vector, N, N)
    # religion
    X???_vector = trade_data_matrix[:,5]
    X??? = reshape(X???_vector, N, N)

    X??? = coalesce.(X???, 0)
    X??? = coalesce.(X???, 0)
    X??? = coalesce.(X???, 0)
    X??? = coalesce.(X???, 0)
    X??? = coalesce.(X???, 0)
    X??? = coalesce.(X???, 0)
    X??? = coalesce.(X???, 0)
    X??? = coalesce.(X???, 0)

    eval(Main, Expr(:delete, :trade_data_dgp))
    eval(Main, Expr(:delete, :trade_data))
    eval(Main, Expr(:delete, :trade_data_matrix))
    eval(Main, Expr(:delete, :coefficients_tobit_label))
    eval(Main, Expr(:delete, :coefficients_tobit))
    eval(Main, Expr(:delete, :scale_tobit))
    eval(Main, Expr(:delete, :sigma_tobit))
    eval(Main, Expr(:delete, :X???_vector))
    eval(Main, Expr(:delete, :X???_vector))
    eval(Main, Expr(:delete, :X???_vector))
    eval(Main, Expr(:delete, :X???_vector))
    eval(Main, Expr(:delete, :X???_vector))
    eval(Main, Expr(:delete, :X???_vector))
    eval(Main, Expr(:delete, :X???_vector))
    eval(Main, Expr(:delete, :X???_vector))
    GC.gc()
end

@everywhere function simulation(nobs, W_matrix, theta_tobit, ys_tobit, n_thres, taus, X???, X???, X???, X???, X???, X???, X???, X???, sim)
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
    ????_comb_grid  = Array{Float64}(undef, n_thres, 8)
    ????_comb_grid .= NaN
    se???_grid = Array{Float64}(undef, n_thres, 8)
    se???_grid .= NaN
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

        index_i_comb, index_j_comb, y??_comb, b_comb, c_comb, nondiag_comb, X?????_comb, X?????_comb, X?????_comb, X?????_comb, X?????_comb, X?????_comb, X?????_comb, X?????_comb =  dataquantilecomb(Y_1, X???, X???, X???, X???, X???, X???, X???, X???)

        X?????_comb_cond = @view X?????_comb[c_comb.==1]
        X?????_comb_cond = @view X?????_comb[c_comb.==1]
        X?????_comb_cond = @view X?????_comb[c_comb.==1]
        X?????_comb_cond = @view X?????_comb[c_comb.==1]
        X?????_comb_cond = @view X?????_comb[c_comb.==1]
        X?????_comb_cond = @view X?????_comb[c_comb.==1]
        X?????_comb_cond = @view X?????_comb[c_comb.==1]
        X?????_comb_cond = @view X?????_comb[c_comb.==1]
        y??_comb_cond = @view y??_comb[c_comb.==1]
        nondiag_comb_cond = @view nondiag_comb[c_comb.==1]

        X??_comb_cond = [X?????_comb_cond X?????_comb_cond X?????_comb_cond X?????_comb_cond X?????_comb_cond X?????_comb_cond X?????_comb_cond X?????_comb_cond]
        eval(Main, Expr(:delete, :X?????_comb_cond))
        eval(Main, Expr(:delete, :X?????_comb_cond))
        eval(Main, Expr(:delete, :X?????_comb_cond))
        eval(Main, Expr(:delete, :X?????_comb_cond))
        eval(Main, Expr(:delete, :X?????_comb_cond))
        eval(Main, Expr(:delete, :X?????_comb_cond))
        eval(Main, Expr(:delete, :X?????_comb_cond))
        eval(Main, Expr(:delete, :X?????_comb_cond))
        GC.gc()

        ?????_comb = zeros(8,1)
        ?????_comb = initialize!(?????_comb, X??_comb_cond, y??_comb_cond)
        nll_comb = make_closures(X??_comb_cond, y??_comb_cond)
        result_comb = optimize(nll_comb, ?????_comb, LBFGS(), autodiff=:forward)
        ????_comb = Optim.minimizer(result_comb)

        ????_comb_grid[g,:] = ????_comb'

        se??? = standarderrors_application(Y_1, X???, X???, X???, X???, X???, X???, X???, X???, ????_comb, X??_comb_cond,nondiag_comb_cond)
        se???_grid[g,:] = se???'

        eval(Main, Expr(:delete, :X??_comb_cond))
        eval(Main, Expr(:delete, :nondiag_comb_cond))
        eval(Main, Expr(:delete, :y??_comb_cond))
        GC.gc()

    end

    eval(Main, Expr(:delete, :Y))
    eval(Main, Expr(:delete, :Ysim))
    eval(Main, Expr(:delete, :Ydummy_matrix))
    GC.gc()

    output = hcat(????_comb_grid, se???_grid, quantile_grid, percentage_nodes, ys_tobit, simulation_number)

    file_path = "results_simulations/simulation_output_della_$(sim).jld2"
    save(file_path, "output", output)

    return output

end

sims = 50
simulation_output_total = @time @showprogress pmap(1:sims) do sim
    simulation(nobs, W_matrix, theta_tobit, ys_tobit, n_thres, taus, X???, X???, X???, X???, X???, X???, X???, X???, sim)
end

save("results_simulations/simulation_output_total_della.jld2","simulation_output_total",simulation_output_total)



#check if passing everything needed inside function