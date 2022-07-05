using DelimitedFiles
using LinearAlgebra
using NamedArrays
using ArgParse
using Random
using CUDA

include("../../modules/NetworkBasedInference.jl/src/NetworkBasedInference.jl")
using .NetworkBasedInference

# Use GPU if available, otherwise CPU
# (CUDA vs BLAS, in short)
if CUDA.functional()
    useGPU(x::AbstractArray) = CuArray{Float32}(x)
else
    useGPU(x::AbstractArray) = x
end

# Node degree functions
k(vᵢ::Integer, G::AbstractMatrix) = count(!iszero, G[vᵢ,:])
k(eᵢ::AbstractVector) = count(!iszero, eᵢ)
k(G::AbstractMatrix) = mapslices(k, G; dims = 2)

function split(G::NamedArray, ngroups::Int64; seed::Int64 = 1)
    E = Tuple.(findall(!iszero,G))

    # Assign fold to edges of graph
    shuffle!(MersenneTwister(seed), E)
    groups = [ [] for _ in 1:ngroups ]

    for (i, eᵢ) in enumerate(E)
        foldᵢ = mod(i, ngroups) + 1
        push!(groups[foldᵢ], eᵢ)
    end

    return groups
end

function createAdjMatrix(DT::AbstractMatrix, DS::AbstractMatrix)
    # Get size of matrices and check if the dimensions match from graph preparation
    Nd₁, Nt = size(DT)
    Nd₂, Ns = size(DS)
    @assert Nd₁ == Nd₂

    # Create matrix A
    Mdd = zeros(Nd₁, Nd₁)
    Mds = DS
    Mdt = DT

    Msd = Mds'
    Mss = zeros(Ns, Ns)
    Mst = zeros(Ns, Nt)

    Mtd = Mdt'
    Mts = Mst'
    Mtt = zeros(Nt, Nt)

    A = [ Mdd Mds Mdt ; 
          Msd Mss Mst ;
          Mtd Mts Mtt ]

    # Create matrix B (A == B in this case!)
    B = deepcopy(A)

    return (A, B)
end

function createAdjMatrix(DT::AbstractMatrix, DS::AbstractMatrix, C::AbstractArray)
    # Get size of matrices and check if the dimensions match from graph preparation
    Nd₁, Nt = size(DT)
    Nd₂, Ns = size(DS)
    Nc      = length(C)
    @assert Nd₁ == Nd₂

    # Create matrix A
    Mcc = zeros(Nc, Nc)
    Mcd = zeros(Nc, Nd₁-Nc)
    Mcs = DS[C,:]
    Mct = zeros(Nc, Nt)

    Mdc = Mcd'
    Mdd = zeros(Nd₁-Nc, Nd₁-Nc)
    Mds = DS[findall(!in(C), 1:end),:]
    Mdt = DT[findall(!in(C), 1:end),:]

    Msc = Mcs'
    Msd = Mds'
    Mss = zeros(Ns, Ns)
    Mst = zeros(Ns, Nt)

    Mtc = Mct'
    Mtd = Mdt'
    Mts = Mst'
    Mtt = zeros(Nt, Nt)

    A = [ Mcc Mcd Mcs Mct ; 
          Mdc Mdd Mds Mdt ; 
          Msc Msd Mss Mst ;
          Mtc Mtd Mts Mtt ]

    # Create matrix B
    B = deepcopy(A)
    B[begin:Nc,:] .= 0
    B[:,begin:Nc] .= 0

    return (A, B)
end

function prepare(DT₀::NamedArray, DS₀::NamedArray, Eᵢ::AbstractArray)
    # Assign names to rows and columns
    Nd, Nt  = size(DT₀)
    _,  Ns = size(DS₀)

    # Create temporal DT graph
    DT₁  = deepcopy(DT₀)
    DT′  = similar(DT₀)
    DT′ .= 0
    for eᵢ in Eᵢ
        s, t = eᵢ
        DT₁[s,t] = 0
        DT′[s,t] = DT₀[s,t]
    end

    # Get drugs that don't have any link with targets (degree == 0)
    # NOTE: we refer to this drugs as "compounds", hence the "C" denotation.
    C  = [ dᵢ for (dᵢ, kᵢ) in enumerate(k(DT₁)) if kᵢ == 0]
    Nc = length(C)

    # Create matrices for SDTNBI prediction
    A₁, B₁ = NamedArray.(createAdjMatrix(DT₁, DS₀))
    A₂, B₂ = NamedArray.(createAdjMatrix(DT₁, DS₀, C))

    # Name indices in matrices
    Names₁  = vcat(["D$i" for i in 1:Nd], 
                   ["S$i" for i in 1:Ns], 
                   ["T$i" for i in 1:Nt])
    Names₂  = vcat(["D$i" for i in C], 
                   ["D$i" for i in 1:Nd if !in(i, C)], 
                   ["S$i" for i in 1:Ns], 
                   ["T$i" for i in 1:Nt])

    for G in [A₁, B₁]
        setnames!(G, Names₁, 1)
        setnames!(G, Names₁, 2)
    end

    for G in [A₂, B₂]
        setnames!(G, Names₂, 1)
        setnames!(G, Names₂, 2)
    end

    return (A₁, A₂), (B₁, B₂), C, DT′
end

function predict(A::NamedArray, B::NamedArray)
    W = NetworkBasedInference.denovoNBI(B)
    R = NamedArray(Matrix(useGPU(A)*useGPU(W)^2))

    for dim in 1:2
        setnames!(R, names(A, dim), dim)
    end

    return R
end

function clean(R::NamedArray, A::NamedArray)
    R′ = deepcopy(R)

    # Get degrees for nodes in bipartite graph A
    Kₐ = k(A)

    # Clean predictions adjacancy matrix R from disconnected targets
    for (tᵢ, k) in zip(names(A,1), Kₐ)
        if k == 0
            R′[:,tᵢ] .= -99
            R′[tᵢ,:] .= -99
        end
    end

    return R′
end

function save(R::NamedArray, DT::NamedArray, E::Array, foldᵢ::Int64, fout::String)
    L = unique(["D$(nₛ)" for (nₛ,nₜ) in E])
    T = [n for n in names(DT,2) if occursin("T",n)]

    open(fout, "a+") do f
        for nₛ in L
            for nₜ in T
                formatted_output = "$foldᵢ, $nₛ, $nₜ, $(R[nₛ, nₜ]), $(DT[nₛ, nₜ])"
                write(f, formatted_output * "\n")
            end
        end 
    end
end

save(R::Nothing, DT::NamedArray, E::Array, foldᵢ::Int64, fout::String) = nothing

function main(args)
    configs = ArgParseSettings()

    add_arg_group!(configs, "I/O options:")
    @add_arg_table! configs begin
        "--DT"
            arg_type = String
            action   = :store_arg
            help     = "Drug-Target adjacency matrix"
            required = true
        "--DS"
            arg_type = String
            action   = :store_arg
            help     = "Drug-Drug similarity matrix"
            required = true
        "--output-file", "-o"
            arg_type = String
            action   = :store_arg
            help     = "Predictions output template name"
            required = true
    end
    add_arg_group!(configs, "CV options:")
    @add_arg_table! configs begin
        "--n-iterations", "-n"
            arg_type = Int64
            action   = :store_arg
            help     = "Number of iterations runned"
            required = false
            default  = 10
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
    end
    add_arg_group!(configs, "Misc options:")
    @add_arg_table! configs begin
        "--gpu-id"
            arg_type = Int64
            action   = :store_arg
            help     = "GPU ID"
            required = false
            default  = 0
    end

    parsed_args = parse_args(args, configs)
    
    # Argument parsing
    template = parsed_args["output-file"]
    niterations = parsed_args["n-iterations"]
    kfolds = parsed_args["k-folds"]
    seed   = parsed_args["seed"]
    
    # Use specific GPU if available, otherwise CPU
    if CUDA.functional()
        device!(parsed_args["gpu-id"])
    end

    # Load matrices to memory
    DT = NamedArray(readdlm(parsed_args["DT"], Float64))
    DS = NamedArray(readdlm(parsed_args["DS"], Float64))

    Nd, Nt = size(DT)
    _, Ns = size(DS)

    setnames!(DT, ["D$i" for i in 1:Nd], 1)
    setnames!(DT, ["T$i" for i in 1:Nt], 2)
    setnames!(DS, ["D$i" for i in 1:Nd], 1)
    setnames!(DS, ["S$i" for i in 1:Ns], 2)

    # k-fold cross-validation procedure
    for iter in 1:niterations
        Eₜ = split(DT, kfolds; seed = seed+iter)
        for foldᵢ in 1:length(Eₜ)
            Eᵢ           = Eₜ[foldᵢ]
            A, B, _, DT′ = prepare(DT, DS, Eᵢ)
            _, A₂        = A
            _, B₂        = B
            R₂           = predict(A₂, B₂)
            R₂′          = clean(R₂, A₂)
            save(R₂′, DT′, Eᵢ, foldᵢ, replace(template, "NN"=>"N$(iter)"))
        end
    end
end

main(ARGS)
