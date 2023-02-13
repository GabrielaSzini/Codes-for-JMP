using ParallelUtilities
using SharedArrays
using Distributions
using ForwardDiff
using StatsFuns
#using GLM
using StatFiles
using DataFrames
#using MATLAB
using DelimitedFiles
using ProgressMeter
using ThreadsX
using Base.Iterators
using Random
using LinearAlgebra
using Optim, NLSolversBase
using Base.Threads
using FileIO, JLD2


using Distributed
addprocs(4)

#TRY LATED WITHOUT FIRST LINE IN EVERYWHERE?
@everywhere begin
    using Pkg; Pkg.activate(".")
    using SharedArrays
    using Distributions
    using ForwardDiff
    using StatsFuns
    #using GLM
    using StatFiles
    using DataFrames
    #using MATLAB
    using DelimitedFiles
    using ProgressMeter
    using ThreadsX
    using Base.Iterators
    using Random
    using LinearAlgebra
    using Optim, NLSolversBase
    using Base.Threads

    include("utils/stats.jl")

    # first we import the data, remembering that we want to have the variables as N*N matrices.
    # remember to exclude Congo as it does not export with none of the countries, we should have then 157 countries
    # we use the year 1986
    # explanatory variables: logarithm of the distance in kilometers between country i’s capital and country j’s capital and indicators for common colonial ties, currency union, regional free trade area (FTA), border, legal system, language, and religion.

    trade_data = DataFrame(load("trade.dta"))
    trade_data_matrix = convert(Matrix, trade_data)
    N=157 
    dimension_reg=8

    # volume of trade
    Y_vector = trade_data_matrix[:,16]
    Y = reshape(Y_vector, N, N)
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

    grid, n_points = gridsquantile(Y, N)

    # after obtaining the grids we can replace missing values with zeros as they do not matter (only diagonal)
    Y = coalesce.(Y, 0)
    X₁ = coalesce.(X₁, 0)
    X₂ = coalesce.(X₂, 0)
    X₃ = coalesce.(X₃, 0)
    X₄ = coalesce.(X₄, 0)
    X₅ = coalesce.(X₅, 0)
    X₆ = coalesce.(X₆, 0)
    X₇ = coalesce.(X₇, 0)
    X₈ = coalesce.(X₈, 0)

    Ydummy_matrix = dummy(Y, N, grid, n_points)

    println("Finalized dataset")

    #for loop from point 55 to 99: need to store 8*2 parameters and 8*4 standard errors
    #first we create the vectors to store results
    Nquantiles = 99-55+1

    println("Finalized initialization")
    total_iterations = 99-54

end

#SAVE THE QUANTILE

β̂_comb_grid  = SharedArray{Float64,2}((Nquantiles,8))
#β̂_comb_grid = Array{Float64}(undef, Nquantiles, 8)
#β̂_perm_grid = Array{Float64}(undef, Nquantiles, 8)
se₁_grid = SharedArray{Float64,2}((Nquantiles,8))
quantile_grid = SharedArray{Float64,2}((Nquantiles,1))
percentage_nodes = SharedArray{Float64,2}((Nquantiles,1))

#se₁_grid = Array{Float64}(undef, Nquantiles, 8)
#se₂_grid = Array{Float64}(undef, Nquantiles, 8)
#se₃_grid = Array{Float64}(undef, Nquantiles, 8)
#se₄_grid = Array{Float64}(undef, Nquantiles, 8)

#make function taking value g, and use slide 24

@time @sync @distributed for g in 55:99
    index = g-54
    quantile_grid[index,1] = g
    print("Iteration $index / $total_iterations \r")

    #taking Y's for the grid point
    Y_1 = Ydummy_matrix[:,:,g]
    percentage_nodes[index,1] = (sum(Y_1) - 157)/(157*156)

    # a in output is equivalent to ỹ
    # point estimates with combinations
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

    β̂_comb_grid[index,:] = β̂_comb'

    println("Finalized first optimization")

    # point estimates with permutations
    #index_i_perm, index_j_perm, ỹ_perm, b_perm, c_perm, nondiag_perm, X̃₁_perm, X̃₂_perm, X̃₃_perm, X̃₄_perm, X̃₅_perm, X̃₆_perm, X̃₇_perm, X̃₈_perm = dataquantileperm(Y_1, X₁, X₂, X₃, X₄, X₅, X₆, X₇, X₈)

    #X̃₁_perm_cond = @view X̃₁_perm[c_perm.==1]
    #X̃₂_perm_cond = @view X̃₂_perm[c_perm.==1]
    #X̃₃_perm_cond = @view X̃₃_perm[c_perm.==1]
    #X̃₄_perm_cond = @view X̃₄_perm[c_perm.==1]
    #X̃₅_perm_cond = @view X̃₅_perm[c_perm.==1]
    #X̃₆_perm_cond = @view X̃₆_perm[c_perm.==1]
    #X̃₇_perm_cond = @view X̃₇_perm[c_perm.==1]
    #X̃₈_perm_cond = @view X̃₈_perm[c_perm.==1]
    #ỹ_perm_cond = @view ỹ_perm[c_perm.==1]

    #X̃_perm_cond = [X̃₁_perm_cond X̃₂_perm_cond X̃₃_perm_cond X̃₄_perm_cond X̃₅_perm_cond X̃₆_perm_cond X̃₇_perm_cond X̃₈_perm_cond]

    #println("Finalized second optimization")

    #β₀_perm = zeros(8,1)
    #β₀_perm = initialize!(β₀_perm, X̃_perm_cond, ỹ_perm_cond)
    #nll_perm = make_closures(X̃_perm_cond, ỹ_perm_cond)
    #result_perm = optimize(nll_perm, β₀_perm, LBFGS(), autodiff=:forward)
    #β̂_perm = Optim.minimizer(result_perm)

    #β̂_perm_grid[index,:] = β̂_perm'

    #standard errors
    se₁ = standarderrors_application(Y_1, X₁, X₂, X₃, X₄, X₅, X₆, X₇, X₈, β̂_comb, X̃_comb_cond,nondiag_comb_cond)
    se₁_grid[index,:] = se₁'

    println("Finalized standard errors")

end

save("results_application/beta_comb_grid.jld2","β̂_comb_grid",β̂_comb_grid)
save("results_application/se_grid.jld2","se₁_grid",se₁_grid)
save("results_application/quantile_grid.jld2","quantile_grid",quantile_grid)
save("results_application/percentage_nodes.jld2","percentage_nodes",percentage_nodes)

betas = FileIO.load("results_application/beta_comb_grid.jld2","β̂_comb_grid")
ses =  FileIO.load("results_application/se_grid.jld2","se₁_grid")
quantile = FileIO.load("results_application/quantile_grid.jld2","quantile_grid")
percentage = FileIO.load("results_application/percentage_nodes.jld2","percentage_nodes")


#DISTANCE = 8 \BETA_1
#LEGAL = 4 \BETA_6

beta_distance = betas[:,1]
se_distance = ses[:,1]
ci_distance_lower = beta_distance .- 1.96.*se_distance
ci_distance_upper = beta_distance .+ 1.96.*se_distance

beta_legal = betas[:,6]
se_legal = ses[:,6]
ci_legal_lower = beta_legal .- 1.96.*se_legal
ci_legal_upper = beta_legal .+ 1.96.*se_legal

plot(quantile, [ci_distance_lower  beta_distance ci_distance_upper], title="Log distance", label=["95% CI" "DR coefficient" "95% CI"], lc=[:blue :black :blue], linewidth=2)
xlabel!("Quantile of trade")
ylabel!("DR coeff")
plot!(legend=:topleft, legendcolumns=3)
savefig("distance_all.png")

plot(quantile[1:40], [ci_distance_lower[1:40]  beta_distance[1:40] ci_distance_upper[1:40]], title="Log distance", label=["95% CI" "DR coefficient" "95% CI"], lc=[:blue :black :blue], linewidth=2)
xlabel!("Quantile of trade")
ylabel!("DR coeff")
plot!(legend=:topleft, legendcolumns=3)
plot!(size=(380,350))
savefig("distance_cut.png")

plot(quantile, [ci_legal_lower  beta_legal ci_legal_upper], title="Legal", label=["95% CI" "DR coefficient" "95% CI"], lc=[:blue :black :blue], linewidth=2)
xlabel!("Quantile of trade")
ylabel!("DR coeff")
plot!(legend=:topleft, legendcolumns=3)
savefig("legal_all.png")

plot(quantile[1:40], [ci_legal_lower[1:40]  beta_legal[1:40] ci_legal_upper[1:40]], title="Legal", label=["95% CI" "DR coefficient" "95% CI"], lc=[:blue :black :blue], linewidth=2)
xlabel!("Quantile of trade")
ylabel!("DR coeff")
plot!(legend=:bottomleft, legendcolumns=3)
plot!(size=(380,350))
savefig("legal_cut.png")

