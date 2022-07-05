#!/usr/local/bin/julia

using ArgParse
using LinearAlgebra
using DelimitedFiles
using Random
using NamedArrays
using CUDA

include("../../modules/NetworkBasedInference.jl/src/NetworkBasedInference.jl")
using .NetworkBasedInference

# Use GPU if available, otherwise CPU
# (CUDA vs BLAS, in short)
if CUDA.functional()
    useGPU(x::AbstractArray) = CuArray(x)
else
    useGPU(x::AbstractArray) = x
end

"""
	split(F₀, kfolds, rng)

Split adjacency matrix into N groups

# Arguments
- `G::AbstractMatrix`: Bipartite network adjacency matrix.
- `ngroups::Integer`: Number of groups to split.
- `seed::Integer`: Splitting seed.
"""
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

"""
	prepare(F₀, E)

Prepare adjacency matrix of bipartite network used in NBI.

# Arguments
- `F₀::AbstractMatrix`: Bipartite network adjacency matrix.
- `E::AbstractVector`:  List of edges to remove from `L₂L₃`.
"""
function prepare(F₀::AbstractMatrix, E::AbstractVector)
	# Create NBI-ready `F₀` adjacency matrix
	F₁   = deepcopy(F₀)
	F₀′  = similar(F₀)
	F₀′ .= 0
	for (s,t) in E
		F₁[s,t]  = 0
		F₀′[s,t] = F₀[s,t]
	end

	# Get isolates in `F₁`
	Kₘ = vec(count(!iszero, F₁; dims=2))			# Get degree vector for row nodes
	Kₙ = vec(count(!iszero, F₁; dims=1))			# Get degree vector for column nodes
	Iₘ = findall(iszero, Kₘ)						# Get indices for isolate row nodes
	Iₙ = findall(iszero, Kₙ)						# Get indices for isolate column nodes
	I  = (Rows = Iₘ, Columns = Iₙ)

	# Filter out isolates of network
	# NOTE: This is done because the implementation of NBI is based in matrix multiplication.
	# BLAS throws an error when either a row or a column is populated exclusively with zeros,
	# therefore we remove this for the prediction algorithm to later on add them back (This 
	# is counterintuitive, but it's the only way I was able to make it work). -CVV
	F₁ = F₁[1:end .∉ [Iₘ], 1:end .∉ [Iₙ]]

	return F₀′, F₁, I
end

"""
	predict(F₀′)

Predict DTI's using NBI algorithm

# Arguments
- `F₀::AbstractMatrix`: Bipartite network adjacency matrix.
"""
function predict(F₀′::AbstractMatrix)
    # Get named indices
	namesₘ = names(F₀′, 1)
	namesₙ = names(F₀′, 2)

    # Calculate resource allocation
	W  = NBI(F₀′)
    F₁ = NamedArray(Matrix(useGPU(F₀′) * useGPU(W)))

    # Name indices
	setnames!(F₁, namesₘ, 1)
	setnames!(F₁, namesₙ, 2)

	return F₁
end

"""
	clean!(R, L₂L₃)

Clean predictions

# Arguments
- `F₁::AbstractMatrix`: Bipartite network predictions matrix.
- `I`: Isolated nodes.
"""
function clean!(F₁::AbstractMatrix, I)
	F₁′ = deepcopy(F₁)
	
	# Add rows for isolated drugs
	# NOTE: This are considered as prediction with score of '0', since we keep this 
	# predictions as if the algorithm is unable to generate de novo predictions. Limitation
	# of NBI, hence we need to acknoledge this in the output.
	for Dᵢ in I[:Rows]
		namesₘ = names(F₁′, 1)				# Get names of row indices
		namesₙ = names(F₁′, 2)				# Get names of columns indices
		push!(namesₘ, "D$Dᵢ")				# Add name for current isolated node processed

		R = NamedArray(zeros(1, length(namesₙ)))
		#R .= -99							# This line makes this predictions ignored by evaluation
		F₁′ = vcat(F₁′, R)
		setnames!(F₁′, namesₘ, 1)
		setnames!(F₁′, namesₙ, 2)
	end
	
	# Add columns for isolated targets
	# NOTE: This are considered as predictions with score of '-99', since we ignore this
	# predictions when evaluating the predictive performance. This is because we do the same
	# for all NBI derived methods in the project.
	for Tᵢ in I[:Columns]
		namesₘ = names(F₁′, 1)				# Get names of row indices
		namesₙ = names(F₁′, 2)				# Get names of columns indices
		push!(namesₙ, "T$Tᵢ")					# Add name for current isolated node processed

		R = NamedArray(zeros(length(namesₘ)))
		R .= -99							# This line makes this predictions ignored by evaluation
		F₁′ = hcat(F₁′, R)
		setnames!(F₁′, namesₘ, 1)
		setnames!(F₁′, namesₙ, 2)
	end
	
	# @info "Isolated nodes" I
	# @info "Nodes in F₁′" names(F₁′)
	return F₁′
end

"""
	save(R, DT, E, fold, fout)

Save predictions as edge-list

# Arguments
- `R::NamedArray`: Bipartite network predictions matrix.
- `DT::NamedArray`: Bipartite network adjacency matrix.
- `E::Array`: Test set edges.
- `foldᵢ::Int64`: Fold number/id.
- `fout::String`: Output file path.
"""
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

function main(args)
	configs = ArgParseSettings()

	add_arg_group!(configs, "I/O options:")
	@add_arg_table! configs begin
		"--DT", "-i"
			arg_type = String
			action	 = :store_arg
			help	 = "Drug-Target adjacency matrix"
			required = true
        "--output-file", "-o"
			arg_type = String
			action	 = :store_arg
			help	 = "Predictions output template name"
			required = true
	end
    add_arg_group!(configs, "CV options:")
    @add_arg_table! configs begin
        "--k-iterations", "-n"
            arg_type = Int64
            action   = :store_arg
            help     = "Number of iterations to run"
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

	# Store arguments to variables
    niterations = parsed_args["k-iterations"]
	kfolds = parsed_args["k-folds"]
	seed = parsed_args["seed"]
	template = parsed_args["output-file"]
    if CUDA.functional()
        device!(parsed_args["gpu-id"])
    end

    # Load matrices to memory
	DT = NamedArray(readdlm(parsed_args["DT"]))
	setnames!(DT, ["D$i" for i in 1:size(DT,1)], 1)
	setnames!(DT, ["T$i" for i in 1:size(DT,2)], 2)

	# Predict drug-target interactions
    for iter in 1:niterations
        Eₜ = split(DT, kfolds; seed = seed+iter)
        for foldᵢ in 1:kfolds
            Eᵢ        = Eₜ[foldᵢ]
            DT′, A, I = prepare(DT, Eᵢ)
            R         = predict(A)
            R′        = clean!(R, I)
            names(R′)
            save(R′, DT′, Eᵢ, foldᵢ, replace(template, "NN"=>"N$(iter)"))
        end
    end
end

main(ARGS)
