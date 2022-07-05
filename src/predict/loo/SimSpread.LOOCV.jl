#!/usr/local/bin/julia

include("../../modules/NetworkBasedInference.jl/src/NetworkBasedInference.jl")

using ArgParse
using LinearAlgebra
using DelimitedFiles
using .NetworkBasedInference
using CUDA

# Use GPU if available, otherwise CPU
# (CUDA vs BLAS, in short)
if CUDA.functional()
    useGPU(x::AbstractArray) = CuArray{Float32}(x)
else
    useGPU(x::AbstractArray) = x
end

function cutoff(Sc, α; variant = :unweighted)
    if variant == :unweighted
        0 < Sc < α ? 1 : 0
    elseif variant == :weighted
        Sc < α ? Sc : 0
    end
end

function prepare(DS::Matrix, DT::Matrix, C::Int)
    @assert size(DS, 1) == size(DT, 1)

    # Get dimensions of network
    Nc = 1
    Nd = size(DT, 1)
    Nt = size(DT, 2)
    Ns = size(DS, 2)

    # Create matrix A
    Mcc = zeros(Nc, Nc)
    Mcd = zeros(1, Nd - 1)
    Mcs = DS[C, :]'
    Mct = zeros(1, Nt)

    Mdc = Mcd'
    Mdd = zeros(Nd - 1, Nd - 1)
    Mds = DS[1:end.!=C, :]
    Mdt = DT[1:end.!=C, :]

    Msc = Mcs'
    Msd = Mds'
    Mss = zeros(Ns, Ns)
    Mst = zeros(Ns, Nt)

    Mtc = Mct'
    Mtd = Mdt'
    Mts = Mst'
    Mtt = zeros(Nt, Nt)

    A = Matrix{Float64}([Mcc Mcd Mcs Mct
        Mdc Mdd Mds Mdt
        Msc Msd Mss Mst
        Mtc Mtd Mts Mtt])

    # Create matrix B
    B = deepcopy(A)
    B[1, :] .= 0
    B[:, 1] .= 0

    return A, B
end

function predict(DD₀, DT, α, fout; variant = :unweighted)
    Nd, Nt = size(DT)
    _, Ns = size(DD₀)

    DD₁ = Matrix{Float32}(cutoff.(DD₀, α; variant = variant))

    for C in 1:1:Nd
        # Predict targets for compound
        A, B = prepare(DD₁, DT, C)
        W = denovoNBI(B)
        F = useGPU(A) * useGPU(W)^2
        R = Array(F[1, Nd+Ns+1:end])

        # Clean predictions of possible errors produced by the data splitting
        DTₐ = A[begin:Nd, Nd+Ns+1:end]
        kₜ_DTₐ = [sum(col) for col in eachcol(DTₐ)]

        for (i, k) in enumerate(kₜ_DTₐ)
            if k == 0
                R[i] = -99
            end
        end

        # Save predictions to file
        open(fout, "a+") do f
            for T in 1:1:Nt
                write(f, "$C, $C, $T, $(R[T]), $(DT[C,T])\n")
            end
        end
    end
end

function newname(str, α)
    str_α = replace(string(α), '.' => "")

    return str * "_" * str_α * ".out"
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
        help = "Predictions output template name"
        required = true
    end
    add_arg_group!(configs, "Parameters options:")
    @add_arg_table! configs begin
        "--weighted", "-w"
        help = "Drug-Drug weighting scheme"
        action = :store_true
        "--cutoff-resolution", "-r"
        arg_type = Float64
        action = :store_arg
        help = "Similarity cutoff step size"
        default = 0.05
        "--cutoff-min"
        arg_type = Float64
        action = :store_arg
        help = "Similarity cutoff minimum value"
        default = 0.0
        "--cutoff-max"
        arg_type = Float64
        action = :store_arg
        help = "Similarity cutoff maximum value"
        default = 1.0
    end
    add_arg_group!(configs, "Misc options:")
    @add_arg_table! configs begin
        "--gpu-id"
        arg_type = Int64
        action = :store_arg
        help = "GPU ID"
        required = false
        default = 0
    end

    parsed_args = parse_args(args, configs)

    # Store arguments to variables
    if CUDA.functional()
        device!(parsed_args["gpu-id"])
    end
    variant = parsed_args["weighted"] == true ? :weighted : :unweighted
    step = parsed_args["resolution"]
    template = parsed_args["output-file"]
    αₘᵢₙ = parsed_args["beta-min"]
    αₘₐₓ = parsed_args["beta-max"]

    # Load matrices to memory
    DT = readdlm(parsed_args["DT"], Float64)
    DD = readdlm(parsed_args["DD"], Float64)

    for α in αₘᵢₙ:step:αₘₐₓ
        outfile = newname(template, α)
        predict(DD, DT, α, outfile; variant = variant)
    end
end

main(ARGS)
