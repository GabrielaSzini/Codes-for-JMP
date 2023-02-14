# first we import the data and then we need to take N as the number of countries
# discretize and dummify the support run for each point in grid one of the logit functions
# after computing the differences of quadruples, need to check the conditions mentioned by Jochmans
# then, apply logit estimation only to the quadruples satisfying the conditions
"""
Replace NaN
"""
function replace_NaN(matrix::Array{T,2}) where T <: Real
    matrix[isnan.(matrix)] .= 0
    return matrix
end

"""
Function for inverse logit
"""
function inverselogit(x)
    y = log(x / (1 - x))
    return y
end

"""
Function that construct grids based on quantiles
"""
# inputs: N (number of countries), Y (matrix of dependent variable)
# we discretize the support of the dependent variable based on quantiles
function gridsquantile(Y,N)
    p = range(0, 1, length=100)|> collect
    n_points = length(p)
    Y_vector = reshape(Y,N*N,1)
    gridsquantile = quantile(filter(!ismissing,vec(Y_vector)), vec(p))
    return gridsquantile, n_points
end

"""
Function that dummify dependent variable
"""
# inputs: Y (matrix of dependent variable), N (number of countries), grid (discretized support), n_points in grid
# function: we dummify altogether for all grid points the dependent variable.
# given a grid point and the trade flow variable Y, we want to dummify.
# take value 1 if smaller or equal to grid point and else, 0.
function dummy(Y, N, grid, n_points)
    Ydummy_matrix = zeros(N, N, n_points)
    for count in 1:n_points
        gridpoint = grid[count]
        # take Y all columns and rows
        Ydummy = (Y.<= gridpoint)
        Ydummy = convert(Array{Float64}, Ydummy)
        Ydummy_matrix[:,:,count] = Ydummy
    end
    return Ydummy_matrix
end

"""
Function that takes double differencing
"""
# inputs: any matrix M (N*N with dyads) and indices for quadruples
function Δ(M, i, j, k, l)
    @inbounds M[i, j] - M[i, k] - M[l, j] + M[l, k]
end

"""
Function F (logistic)
"""
function F(e)
    F = 1/ (1+ exp(-e))

    return F
end

"""
Function f (logistic)
"""
function f(e)
    f = F(e)*(1 - F(e))
    
    return f
end

"""
Function ff (logistic)
"""
function ff(e)
    ff = f(e)* (1- F(e)) - f(e) * F(e)

    return ff
end

"""
Computing X̃, a, b, c for a given quantile
"""
function dataquantilecomb(Y, X₁, X₂, X₃, X₄, X₅, X₆, X₇, X₈)
    N = size(Y,1)
    nn = binomial(N,2)
    mm = binomial(N-2,2)
    mn = nn * mm #number of combinations
    
    #initializing vectors
    c = Array{Int64}(undef,mn, )
    a = Array{Int64}(undef,mn, )
    b = Array{Int64}(undef,mn, )
    X̃₁ = Array{Float64}(undef,mn, )
    X̃₂ = Array{Float64}(undef,mn, )
    X̃₃ = Array{Float64}(undef,mn, )
    X̃₄ = Array{Float64}(undef,mn, )
    X̃₅ = Array{Float64}(undef,mn, )
    X̃₆ = Array{Float64}(undef,mn, )
    X̃₇ = Array{Float64}(undef,mn, )
    X̃₈ = Array{Float64}(undef,mn, )
    index_i = Array{Int64}(undef,mn, )
    index_j = Array{Int64}(undef,mn, )
    nondiag = Array{Int64}(undef,mn, )

    counter = 1

    @inbounds for i1 in 1:N, j1 in 1:N
        (i1 - j1) == 0 && continue
        for i2 in (i1+1):N
            (i2-i1) * (i2-j1) == 0 && continue
            for j2 in (j1+1):N
                (j2-i1) * (j2-j1) * (j2-i2) == 0 && continue

                nondiag[counter] = (j1≠i1 && j1≠i2 ? 1 : 0) + (j2≠i1 && j2≠i2 ? 1 : 0)

                c[counter] = (( Y[i1, j1] > Y[i1, j2]) && ( Y[i2, j1] <  Y[i2, j2])) || (( Y[i1, j1] < Y[i1, j2]) && ( Y[i2, j1] > Y[i2, j2])) ? 1 : 0
                a[counter] = (( Y[i1, j1] > Y[i1, j2]) && ( Y[i2, j1] <  Y[i2, j2])) ? 1 : 0
                b[counter] = (( Y[i1, j1] < Y[i1, j2]) && ( Y[i2, j1] >  Y[i2, j2])) ? 1 : 0
                
                X̃₁[counter] = Δ(X₁, i1, j1, j2, i2)
                X̃₂[counter] = Δ(X₂, i1, j1, j2, i2)
                X̃₃[counter] = Δ(X₃, i1, j1, j2, i2)
                X̃₄[counter] = Δ(X₄, i1, j1, j2, i2)
                X̃₅[counter] = Δ(X₅, i1, j1, j2, i2)
                X̃₆[counter] = Δ(X₆, i1, j1, j2, i2)
                X̃₇[counter] = Δ(X₇, i1, j1, j2, i2)
                X̃₈[counter] = Δ(X₈, i1, j1, j2, i2)

                index_i[counter] = i1
                index_j[counter] = j1

                counter += 1
            end
        end
    end
    
    # in a row there are the values for one combination
    return index_i, index_j, a, b, c, nondiag, X̃₁, X̃₂, X̃₃, X̃₄, X̃₅, X̃₆, X̃₇, X̃₈

end

"""
Log-likelihood function
"""
function log_likelihood(X̃, ỹ, β)
    ll = .0
    @inbounds for i in eachindex(ỹ)
        zᵢ = dot(X̃[i,:],β)
        c = -log1pexp(-zᵢ)
        ll += ỹ[i] * c + (1-ỹ[i]) * (-zᵢ + c)
    end

    return ll

end

"""
Make a closure for log-likelihood, returning negative
"""
make_closures(X̃, ỹ) = β -> -log_likelihood(X̃, ỹ, β)

"""
Initialization of parameters for log-likelihood
"""
function initialize!(β₀, X̃, ỹ, ϵ = 0.1)
    logit_y = [ifelse(y_i == 1.0, logit(1-ϵ), logit(ϵ)) for y_i in ỹ]
    for j in 1:length(β₀)
        β₀[j] = cov(logit_y, @view(X̃[:,j]))/var(@view(X̃[:,j]))
    end
    
    return β₀

end

"""
Computing X̃, a, b, c for a given quantile considering all permutations
"""
function dataquantileperm(Y, X₁, X₂, X₃, X₄, X₅, X₆, X₇, X₈)
    N = size(Y,1)
    #nn = binomial(N,2)
    #mm = binomial(N-2,2)
    mn = N * (N-1) * (N-2) * (N-3) #number of combinations
    
    #initializing vectors
    c = Array{Int64}(undef,mn, )
    a = Array{Int64}(undef,mn, )
    b = Array{Int64}(undef,mn, )
    X̃₁ = Array{Float64}(undef,mn, )
    X̃₂ = Array{Float64}(undef,mn, )
    X̃₃ = Array{Float64}(undef,mn, )
    X̃₄ = Array{Float64}(undef,mn, )
    X̃₅ = Array{Float64}(undef,mn, )
    X̃₆ = Array{Float64}(undef,mn, )
    X̃₇ = Array{Float64}(undef,mn, )
    X̃₈ = Array{Float64}(undef,mn, )
    index_i = Array{Int64}(undef,mn, )
    index_j = Array{Int64}(undef,mn, )
    nondiag = Array{Int64}(undef,mn, )

    counter = 1

    @inbounds for i1 in 1:N, j1 in 1:N
        (i1 - j1) == 0 && continue
        for i2 in 1:N
            (i2-i1) * (i2-j1) == 0 && continue
            for j2 in 1:N
                (j2-i1) * (j2-j1) * (j2-i2) == 0 && continue

                nondiag[counter] = (j1≠i1 && j1≠i2 ? 1 : 0) + (j2≠i1 && j2≠i2 ? 1 : 0)

                c[counter] = ((Y[i1, j1] > Y[i1, j2]) && (Y[i2, j1] < Y[i2, j2])) || (( Y[i1, j1] < Y[i1, j2]) && (Y[i2, j1] > Y[i2, j2])) ? 1 : 0
                a[counter] = ((Y[i1, j1] > Y[i1, j2]) && (Y[i2, j1] < Y[i2, j2])) ? 1 : 0
                b[counter] = ((Y[i1, j1] < Y[i1, j2]) && (Y[i2, j1] > Y[i2, j2])) ? 1 : 0
                
                X̃₁[counter] = Δ(X₁, i1, j1, j2, i2)
                X̃₂[counter] = Δ(X₂, i1, j1, j2, i2)
                X̃₃[counter] = Δ(X₃, i1, j1, j2, i2)
                X̃₄[counter] = Δ(X₄, i1, j1, j2, i2)
                X̃₅[counter] = Δ(X₅, i1, j1, j2, i2)
                X̃₆[counter] = Δ(X₆, i1, j1, j2, i2)
                X̃₇[counter] = Δ(X₇, i1, j1, j2, i2)
                X̃₈[counter] = Δ(X₈, i1, j1, j2, i2)

                index_i[counter] = i1
                index_j[counter] = j1

                counter += 1
            end
        end
    end

    return index_i, index_j, a, b, c, nondiag, X̃₁, X̃₂, X̃₃, X̃₄, X̃₅, X̃₆, X̃₇, X̃₈

end

function standarderrors_application(Y_1, X₁, X₂, X₃, X₄, X₅, X₆, X₇, X₈, β̂_comb, X̃_comb_cond,nondiag_comb_cond)
    N = size(Y_1,1)
    #nn = binomial(N,2)
    #mm = binomial(N-2,2)
    mn = N * (N-1) * (N-2) * (N-3) #number of combinations
    
    #initializing vectors
    S₁ = zeros(N,N,8)
    #S₃ = zeros(N,N,8)
    J₁ = zeros(8,8)
    #J₂ = zeros(8,8)
    #J₃ = zeros(8,8)
    #J₄ = zeros(8,8)

    @inbounds for i1 in 1:N, j1 in 1:N
        (i1 - j1) == 0 && continue
        y11 = Y_1[i1,j1]
        x₁_11 = X₁[i1,j1]
        x₂_11 = X₂[i1,j1]
        x₃_11 = X₃[i1,j1]
        x₄_11 = X₄[i1,j1]
        x₅_11 = X₅[i1,j1]
        x₆_11 = X₆[i1,j1]
        x₇_11 = X₇[i1,j1]
        x₈_11 = X₈[i1,j1]
        for i2 in 1:N
            (i2-i1) * (i2-j1) == 0 && continue
            y21 = Y_1[i2,j1]
            x₁_21 = X₁[i2,j1]
            x₂_21 = X₂[i2,j1]
            x₃_21 = X₃[i2,j1]
            x₄_21 = X₄[i2,j1]
            x₅_21 = X₅[i2,j1]
            x₆_21 = X₆[i2,j1]
            x₇_21 = X₇[i2,j1]
            x₈_21 = X₈[i2,j1]
            for j2 in 1:N
                (j2-i1) * (j2-j1) * (j2-i2) == 0 && continue
                y12 = Y_1[i1,j2]
                x₁_12 = X₁[i1,j2]
                x₂_12 = X₂[i1,j2]
                x₃_12 = X₃[i1,j2]
                x₄_12 = X₄[i1,j2]
                x₅_12 = X₅[i1,j2]
                x₆_12 = X₆[i1,j2]
                x₇_12 = X₇[i1,j2]
                x₈_12 = X₈[i1,j2]
                y22 = Y_1[i2,j2]
                x₁_22 = X₁[i2,j2]
                x₂_22 = X₂[i2,j2]
                x₃_22 = X₃[i2,j2]
                x₄_22 = X₄[i2,j2]
                x₅_22 = X₅[i2,j2]
                x₆_22 = X₆[i2,j2]
                x₇_22 = X₇[i2,j2]
                x₈_22 = X₈[i2,j2]

                nondiag = (j1≠i1 && j1≠i2 ? 1 : 0) + (j2≠i1 && j2≠i2 ? 1 : 0)

                c = ((y11 > y12) && (y21 < y22)) || ((y11 < y12) && (y21 > y22)) ? 1 : 0
                if c == 0
                    for d in 1:8
                        S₁[i1,j1,d] += c
                        #S₃[i1,j1,d] += X̃[d] * inner2 * c / ((N-2)*(N-3))
                    end

                    for d in 1:8
                        for dd in 1:8
                            J₁[d,dd] +=  c
                         #   J₃[d,dd] +=  X̃[d] * X̃[dd] * inner3₃
                        end
                    end

                else

                    a = ((y11 > y12) && (y21 <  y22)) ? 1 : 0
                    b = ((y11 < y12) && (y21 >  y22)) ? 1 : 0
    
                    X̃₁ = (x₁_11-x₁_12)-(x₁_21-x₁_22)
                    X̃₂ = (x₂_11-x₂_12)-(x₂_21-x₂_22)
                    X̃₃ = (x₃_11-x₃_12)-(x₃_21-x₃_22)
                    X̃₄ = (x₄_11-x₄_12)-(x₄_21-x₄_22)
                    X̃₅ = (x₅_11-x₅_12)-(x₅_21-x₅_22)
                    X̃₆ = (x₆_11-x₆_12)-(x₆_21-x₆_22)
                    X̃₇ = (x₇_11-x₇_12)-(x₇_21-x₇_22)
                    X̃₈ = (x₈_11-x₈_12)-(x₈_21-x₈_22)
    
                    #double check if X is a column vector
                    X̃ = [X̃₁, X̃₂, X̃₃, X̃₄, X̃₅, X̃₆, X̃₇, X̃₈]
    
                    inner = X̃'*β̂_comb
                    inner2 = a - F(inner[1])
    
                    for d in 1:8
                        S₁[i1,j1,d] += 4 * X̃[d] * inner2 * c * nondiag./ ((N-2)*(N-3))
                        #S₃[i1,j1,d] += X̃[d] * inner2 * c / ((N-2)*(N-3))
                    end
    
                    inner3₁ = (0 - f(inner[1]))* c * nondiag / mn
                    #inner3₃ = (0 - f(inner[1]))* c / mn
    
                    for d in 1:8
                        for dd in 1:8
                            J₁[d,dd] +=  X̃[d] * X̃[dd] * inner3₁
                         #   J₃[d,dd] +=  X̃[d] * X̃[dd] * inner3₃
                        end
                    end
                end
            end
        end
    end

    V₁ = zeros(8,8)
    #V₃ = zeros(8,8)

    for d in 1:8
        for dd in 1:8
            V₁[d,dd] = mean(S₁[:,:,d].*S₁[:,:,dd])
            #V₃[d,dd] = mean(S₃[:,:,d].*S₃[:,:,dd])
        end
    end

    W₁ = inv(J₁'*J₁)'*(J₁'*V₁*J₁)*inv(J₁'*J₁)
    se₁ = sqrt.(diag(W₁)./(N*N))

    #W₃ = inv(J₃'*J₃)'*(J₃'*V₃*J₃)*inv(J₃'*J₃)
    #se₃ = sqrt.(diag(W₃)./(N*N))

    #including only combinations in the Hessian
    #nn = binomial(N,2)
    #mm = binomial(N-2,2)
    #ρ = nn * mm #number of combinations

    #inner_comb = X̃_comb_cond*β̂
    #inner2_comb₂ = (0 .- f.(inner_comb)) .* nondiag_comb_cond ./ ρ
    #inner2_comb₄ = (0 .- f.(inner_comb)) ./ ρ  
    
    #for d in 1:8
    #    for dd in 1:8
    #        J₂[d,dd] = sum(X̃_comb_cond[:,d].*X̃_comb_cond[:,dd].*inner2_comb₂)
    #        J₄[d,dd] = sum(X̃_comb_cond[:,d].*X̃_comb_cond[:,dd].*inner2_comb₄)
    #    end
    #end

    #V₂ = V₁
    #V₄ = V₃

    #W₂ = inv(J₂'*J₂)'*(J₂'*V₂*J₂)*inv(J₂'*J₂)
    #se₂ = sqrt.(diag(W₂)./(N*N))

    #W₄ = inv(J₄'*J₄)'*(J₄'*V₄*J₄)*inv(J₄'*J₄)
    #se₄ = sqrt.(diag(W₄)./(N*N))

    return se₁

end

"""
Winsorize variable: only upper quantile
"""
function winsorize(x, upper_quantile)
    upper_threshold = quantile(x, upper_quantile)
    x_winsorized = x
    x_winsorized[x_winsorized .> upper_threshold] = upper_threshold
    return x_winsorized
end

