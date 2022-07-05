#!/usr/local/bin/julia

using ArgParse
using LinearAlgebra
using DelimitedFiles
using CUDA

# Use GPU if available, otherwise CPU
# (CUDA vs BLAS, in short)
if CUDA.functional()
    useGPU(x::AbstractArray) = CuArray(x)
else
    useGPU(x::AbstractArray) = x
end

"""
    prepare(DD::AbstractMatrix, DT::AbstractMatrix, E::AbstractArray)

Prepare matrices to use in cross-validation scheme. Here we eliminate all the edges from the
test set (denoted with `E`) and delete self-loops for the drugs considered in the matrices.

# Arguments
- `DD::AbstractMatrix`: Drug-Drug similarity matrix.
- `DT::AbstractMatrix`: Drug-Target rectangular adjacency matrix.
- `C::Int64`: Drug number or ID to use as test set.
"""
function prepare(DD::Matrix, DT::Matrix, C::Int64)
    # Remove self-loops in DD
    DD′ = deepcopy(DD)
    DD′[diagind(DD′)] .= 0

    # Remove all drug-target interactions for given compounds (C)
    DT′ = deepcopy(DT)
    DT′[C, :] .= 0

    return DD′, DT′
end

"""
    predict(DD::AbstractMatrix, DT::AbstractMatrix, C::Int64)

Predict drug-target interactions using 1-nearest-neighbour algorithm for the given drug, using
the `DD` similarity matrix as guideline.

# Arguments
- `DD::AbstractMatrix`: Drug-Drug similarity matrix.
- `DT::AbstractMatrix`: Drug-Target rectangular adjacency matrix.
- `C::Int64`: Drug number or ID to use as test set.

# Implementation

Here we use the similarity matrix instead of a distance matrix in the nearest-neighbour
algorithm, therefore in order to eliminate overhead produces by element-wise conversion of
a similarity metric to a distance metric and the lack of a standarized method for given
conversion, we search for the most similar neighbour, in other words, we search for the
farthest neighbour in the `DD` matrix.
"""
function predict(DD′::Matrix, DT′::Matrix, C::Int64)
    # Predict targets for compound
    F₁ = useGPU(DD′[:, C]) .* useGPU(DT′)
    R = mapslices(maximum, F₁; dims = 1)

    return R
end

"""
    clean!(R::AbstractMatrix, DT::AbstractMatrix)

Flag all drug-target interactions predictions that weren't able to be predicted by the
method because of limitations in the data splitting procedure. The flag is harcoded to be
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
function clean!(R::AbstractMatrix, DT::AbstractMatrix)
    # Get degree for all targets in `DT` adjacency matrix
    DT_Kₜ = [sum(col) for col in eachcol(DT)]

    # Flag predictions for all targets with degree == 1
    # NOTE: This are the cases that are imposible to predict due to data splitting
    # limitations, therefore we need to ignore them.
    for (i, k) in enumerate(DT_Kₜ)
        if k == 0
            @warn "Target #$i becomes disconnected in data splitting, flagging predictions."
            R[i] = -99
        end
    end
end

"""
    save(R::AbstractMatrix, DT::AbstractMatrix, C::Int64, fout::String)

Save the drug-target interactions predictions as a CSV (comma separated value) file.

# Arguments
- `R::AbstractMatrix`: Predicted drug-target rectangular adjacency matrix.
- `DT::AbstractMatrix`: Drug-Target rectangular adjacency matrix.
- `C::Int64`: Drug number or ID to use as test set.
- `fout::String`: File path of output CSV file.
"""
function save(R::AbstractMatrix, DT::AbstractMatrix, C::Int64, fout::String)
    Nt = length(R)

    # Save predictions to file
    open(fout, "a+") do f
        for T in 1:Nt
            write(f, "$C, $C, $T, $(R[T]), $(DT[C,T])\n")
        end
    end
end

function main(args)
    configs = ArgParseSettings()

    add_arg_group!(configs, "I/O options:")
    @add_arg_table! configs begin
        "--DT"
        arg_type = String
        action = :store_arg
        help = "Drug-Target adjacency matrix"
        required = true
        "--DD"
        arg_type = String
        action = :store_arg
        help = "Drug-Drug similarity matrix"
        required = true
        "--output-file", "-o"
        arg_type = String
        action = :store_arg
        help = "File path for predictions"
        required = true
    end

    parsed_args = parse_args(args, configs)

    # Store arguments to variables
    fout = parsed_args["output-file"]

    # Load matrices to memory
    DT = readdlm(parsed_args["DT"], Float32)
    DD = readdlm(parsed_args["DD"], Float32)

    # Predict drug-target interactions
    Nd = size(DT, 1)
    for C in 1:Nd
        DD′, DT′ = prepare(DD, DT, C)
        R = predict(DD′, DT′, C)
        clean!(R, DT)
        save(R, DT, C, fout)
    end
end

main(ARGS)
