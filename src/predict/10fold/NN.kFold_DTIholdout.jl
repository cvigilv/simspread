#!/usr/local/bin/julia

using ArgParse
using LinearAlgebra
using DelimitedFiles
using Random

k(vᵢ::Integer, G::AbstractMatrix) = count(!iszero, G[vᵢ,:])
k(eᵢ::AbstractVector) = count(!iszero, eᵢ)
k(G::AbstractMatrix) = mapslices(k, G; dims = 2)

"""
    split(DT::AbstractMatrix, k::Int64, rng::Int64)

Split all possible `D` into `k` groups for cross-validation.

# Arguments
- `DT::AbstractMatrix`: Drug-Target rectangular adjacency matrix.
- `k::Int64`: Number of groups to use in data splitting.
- `rng::Int64`: Seed used for data splitting.

"""
function split(G::AbstractMatrix, ngroups::Int64; seed::Int64 = 1)
    # Get array of drugs in adjacency matrix
    Nd = size(G,1)
    D = Array(1:Nd)

    # Assign fold to edges of graph
    shuffle!(MersenneTwister(seed), D)
    groups = [ [] for _ in 1:ngroups ]

    for (i, dᵢ) in enumerate(D)
        foldᵢ = mod(i, ngroups) + 1
        append!(groups[foldᵢ], [(dᵢ,j) for j in findall(!iszero, G[dᵢ,:])]) # TODO(Carlos): Convertir a funcion Γ / neighbours
    end

    return groups
end

"""
    prepare(DD::AbstractMatrix, DT::AbstractMatrix, E::AbstractArray)

Prepare matrices to use in cross-validation scheme. Here we eliminate all the edges from the
test set (denoted with `E`) and delete self-loops for the drugs considered in the matrices.

# Arguments
- `DD::AbstractMatrix`: Drug-Drug similarity matrix.
- `DT::AbstractMatrix`: Drug-Target rectangular adjacency matrix.
- `E::AbstractVector`: Test set edges to delete from `DT`.
"""
function prepare(DD::AbstractMatrix, DT::AbstractMatrix, Eᵢ::AbstractVector)
    # Remove self-loops in DD
    DD′ = deepcopy(DD)
    DD′[diagind(DD′)] .= 0

    # Remove all drug-target interactions for given compounds (C)
    DT₁  = deepcopy(DT)
    DT′  = similar(DT)
    DT′ .= 0
    for eᵢ in Eᵢ
        s, t = eᵢ
        DT₁[s,t] = 0
        DT′[s,t] = DT[s,t]
    end

    return DD′, DT′, DT₁
end

"""
    predict(DD::AbstractMatrix, DT::AbstractMatrix)

Predict all possible drug-target interactions using 1-nearest-neighbour algorithm, using as
distance matrix the `DD` similarity matrix.

# Arguments
- `DD::AbstractMatrix`: Drug-Drug similarity matrix.
- `DT::AbstractMatrix`: Drug-Target rectangular adjacency matrix.

# Implementation

Here we use the similarity matrix instead of a distance matrix in the nearest-neighbour
algorithm, therefore in order to eliminate overhead produces by element-wise conversion of
a similarity metric to a distance metric and the lack of a standarized method for given
conversion, we search for the most similar neighbour, in other words, we search for the
farthest neighbour in the `DD` matrix.
"""
function predict(DD::AbstractMatrix, DT::AbstractMatrix)
    Nd, Nt = size(DT)
    R      = zeros(Nd, Nt)                          # Initialize matrix for storing results

    for C in 1:Nd                                   # Calculate Nearest-Neighbour per drug
        F₁     = DD[:,C] .* DT
        R[C,:] = mapslices(maximum, F₁; dims = 1)
    end

    return R
end

"""
    clean(R::AbstractMatrix, DT::AbstractMatrix)

Flag all drug-target interactions predictions that weren't able to be predicted by the
method because of limitations in the data splitting procedure. The flag is hardcoded to be 
the value `-99`.

# Arguments
- `R::AbstractMatrix`: Predicted drug-target rectangular adjacency matrix.
- `DT::AbstractMatrix`: Drug-Target rectangular adjacency matrix.

# Implementation

Drug-Target interaction dataset are frequently sparse, therefore there is a chance that we
will encounter targets that have only one annotated interaction with drugs. This targets
will be splitted incorrectly in the data splitting procedure, producing predictions with a 
score of 0 (zero) when in reality they can't be predicted. This cases are flagged in order
to exclude them from evaluation.
"""
function clean(R::AbstractArray, DT::AbstractArray)
    R′ = deepcopy(R)

    # Get degrees for nodes in bipartite graph A
    Kₜ = k(DT')

    # Flag predictions for all targets with degree == 0
    # NOTE: This are the cases that are imposible to predict due to data splitting
    # limitations, therefore we need to ignore them.
    for (tᵢ, k) in enumerate(Kₜ)
        if k == 0
            # @warn "Target #$(tᵢ) becomes disconnected in data splitting, flagging predictions."
            R′[:,tᵢ] .= -99
        end
    end

    return R′
end

"""
    save(R::AbstractMatrix, DT::AbstractMatrix, E::AbstractVector, fold::Int64, fout::String)

Save the drug-target interactions predictions as a CSV (comma separated value) file.

# Arguments
- `R::AbstractMatrix`: Predicted drug-target rectangular adjacency matrix.
- `DT::AbstractMatrix`: Drug-Target rectangular adjacency matrix.
- `E::AbstractVector`: Test set edges deleted from `DT`.
- `fold::Int64`: Group number.
- `fout::String`: File path of output CSV file.
"""
function save(R::AbstractMatrix, DT::AbstractMatrix, E::AbstractArray, foldᵢ::Int64, fout::String)
    L = unique([nₛ for (nₛ,nₜ) in E])
    T = 1:size(DT,2)

    open(fout, "a+") do f
        for nₛ in L
            for nₜ in T
                formatted_output = "$foldᵢ, $nₛ, $nₜ, $(R[nₛ, nₜ]), $(DT[nₛ, nₜ])"
                write(f, formatted_output * "\n")
            end
        end 
    end
end

function main(args)
    configs = ArgParseSettings()

    add_arg_group!(configs, "I/O options:")
    @add_arg_table! configs begin
        "--DT"
            arg_type = String
            action   = :store_arg
            help     = "Drug-Target adjacency matrix"
            required = true
        "--DD"
            arg_type = String
            action   = :store_arg
            help     = "Drug-Drug similarity matrix"
            required = true
        "--k-folds", "-k"
            arg_type = Int64
            action   = :store_arg
            help     = "Number of folds used in data splitting"
            required = false
            default  = 10
        "--seed", "-s"
            arg_type = Int64
            action   = :store_arg
            help     = "Seed used for data splitting"
            required = false
            default  = 1
        "--k-iterations", "-n"
            arg_type = Int64
            action   = :store_arg
            help     = "Number of iterations"
            required = false
            default  = 10
        "--output-file", "-o"
            arg_type = String
            action   = :store_arg
            help     = "File path for predictions"
            required = true
    end

    parsed_args = parse_args(args, configs)

    # Store arguments to variables
    kfolds     = parsed_args["k-folds"]
    seed       = parsed_args["seed"]
    iterations = parsed_args["k-iterations"]
    fout       = parsed_args["output-file"]

    # Load matrices to memory
    DT = readdlm(parsed_args["DT"])
    DD = readdlm(parsed_args["DD"])

    # Predict drug-target interactions
    for iter in 1:iterations
        Eₜ = split(DT, kfolds; seed = seed+iter)
        @time for foldᵢ in 1:length(Eₜ)
            Eᵢ = Eₜ[foldᵢ]
            DD′, DT′, DT₁ = prepare(DD, DT, Eᵢ)
            R  = predict(DD′, DT₁)
            R′ = clean(R, DT₁)
            save(R′, DT′, Eᵢ, foldᵢ, replace(fout, "NN"=>"N$(iter)"))
        end
    end
end

main(ARGS)
