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
    useGPU(x::AbstractArray) = CuArray(x)
else
    useGPU(x::AbstractArray) = x
end

function cutoff(Sc, α,; variant = :unweighted)
    if variant == :unweighted
        0 < Sc < α ? 1 : 0
    elseif variant == :weighted
        Sc < α ? Sc : 0
    end
end

function prepare(DTs, DSs)
    # Unpack matrices tuples
    DT₀, DT₁ = DTs
    DS₀, DS₁ = DSs

    # Get dimensions of network
    Nd = size(DT₀,1)
    Nt = size(DT₀,2)
    Ns = size(DS₀,2)
    Nc = size(DS₁,1)

    Mcc = zeros(Nc, Nc)
    Mcd = zeros(Nc, Nd)
    Mcs = DS₁
    Mct = zeros(Nc, Nt)

    Mdc = Mcd'
    Mdd = zeros(Nd, Nd)
    Mds = DS₀
    Mdt = DT₀

    Msc = Mcs'
    Msd = Mds'
    Mss = zeros(Ns, Ns)
    Mst = zeros(Ns, Nt)

    Mtc = Mct'
    Mtd = Mdt'
    Mts = Mst'
    Mtt = zeros(Nt, Nt)

    # Create matrix A and B
    A = Matrix{Float32}([ Mcc Mcd Mcs Mct ;
                          Mdc Mdd Mds Mdt ;
                          Msc Msd Mss Mst ;
                          Mtc Mtd Mts Mtt ])

    B = deepcopy(A)
    B[begin:Nc, :] .= 0
    B[:, begin:Nc] .= 0

    return A, B
end

function predict(DTs, DDs, α, path; variant = :unweighted)
    # Unpack arguments, get dimensions and filter edges between ligands
    DT₀, DTΔ = DTs
    DD₀, DDΔ = DDs

    Nd = size(DT₀, 1)
    Nt = size(DT₀, 2)
    Ns = size(DD₀, 2)
    Nc = size(DDΔ, 1)

    DD₀′ = cutoff.(DD₀, α; variant = variant)
    DDΔ′ = cutoff.(DDΔ, α; variant = variant)

    # Prepare network and predict
    A, B = prepare( (DT₀,DTΔ), (DD₀′,DDΔ′) )
    W    = denovoNBI(B)
    F    = useGPU(A)*useGPU(W)^2                # Send matrix multiplication to GPU
    R    = Matrix(F[begin:Nc, Nc+Nd+Ns+1:end])  # Get product matrix from GPU

    # Save predictions
    open(path, "a+") do f
        for Cᵢ in 1:Nc, Tᵢ in 1:Nt
            write(f, "$Cᵢ, $Cᵢ, $Tᵢ, $(R[Cᵢ,Tᵢ]), $(DTΔ[Cᵢ,Tᵢ])\n")
        end
    end
end

function newname(template, α)
    α′ = replace(string(α), '.' => "")
    return "$(template)_$(α′).out"
end

function main(args)
    # Argument parsing
    configs = ArgParseSettings()

    add_arg_group!(configs, "Options:")
    @add_arg_table! configs begin
        "--DT-current"
            help     = "Current Drug-Target adjacency matrix"
            required = true
            action   = :store_arg
            arg_type = String
        "--DT-future"
            help     = "Future Drug-Target adjacency matrix"
            required = true
            action   = :store_arg
            arg_type = String
        "--DD-current"
            help     = "Current Drug-Drug similarity matrix"
            required = true
            action   = :store_arg
            arg_type = String
        "--DD-future"
            help     = "Future Drug-Drug similarity matrix"
            required = true
            action   = :store_arg
            arg_type = String
        "--weighted", "-w"
            help     = "Drug-Drug weighting scheme"
            action   = :store_true
        "--cutoff-resolution", "-r"
            help     = "Cutoff resolution"
            action   = :store_arg
            arg_type = Float64
            default  = 0.05
        "--cutoff-min"
            arg_type = Float64
            action   = :store_arg
            help     = "Similarity cutoff lower bound"
            default = 0.0
        "--cutoff-max"
            arg_type = Float64
            action   = :store_arg
            help     = "Similarity cutoff top bound"
            default  = 1.0
        "--output-file", "-o"
            help     = "Prediction file name template"
            required = true
            action   = :store_arg
            arg_type = String
    end

    parsed_args = parse_args(args, configs)

    # Store arguments to variables
    variant    = parsed_args["weighted"] == true ? :weighted : :unweighted
    step     = parsed_args["resolution"]
    template   = parsed_args["output-file"]
    αₘᵢₙ = parsed_args["cutoff-min"]
    αₘₐₓ = parsed_args["cutoff-max"]

    println(variant)

    # Load matrices to memory
    DT₀    = readdlm(parsed_args["DT-current"])
    DD₀    = readdlm(parsed_args["DD-current"])
    DTΔ    = readdlm(parsed_args["DT-future"])
    DDΔ    = readdlm(parsed_args["DD-future"])

    for α in αₘᵢₙ:step:αₘₐₓ
        outfile = newname(template, α)
        predict(
            (DT₀,DTΔ),
            (DD₀, DDΔ),
            α,
            outfile;
            variant = variant)
    end
end

main(ARGS)
