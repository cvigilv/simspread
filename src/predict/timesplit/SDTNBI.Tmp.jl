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
    
function predict(DTs, DSs, path)
	# Unpack arguments, get dimensions and filter edges between ligands
	DT₀, DTΔ = DTs
	DS₀, DSΔ = DSs

	@show Nd = size(DT₀, 1)
	@show Nt = size(DT₀, 2)
	@show Ns = size(DS₀, 2)
	@show Nc = size(DSΔ, 1)

	# Prepare network and predict
	A, B = prepare( (DT₀,DTΔ), (DS₀,DSΔ) )
	W    = denovoNBI(B)
	F    = useGPU(A)*useGPU(W)^2				# Send matrix multiplication to GPU
	R    = Matrix(F[begin:Nc, Nc+Nd+Ns+1:end])	# Get product matrix from GPU

	# Save predictions
	open(path, "a+") do f
		for Cᵢ in 1:Nc
			for Tᵢ in 1:Nt
				write(f, "$Cᵢ, $Cᵢ, $Tᵢ, $(R[Cᵢ,Tᵢ]), $(DTΔ[Cᵢ,Tᵢ])\n")
			end
		end
	end
end

function main(args)
	# Argument parsing
	configs = ArgParseSettings()
	
	add_arg_group!(configs, "Options:")
	@add_arg_table! configs begin
		"--DT-current"
			help	 = "Current Drug-Target adjacency matrix"
			required = true
			action	 = :store_arg
			arg_type = String
		"--DT-future"
			help	 = "Future Drug-Target adjacency matrix"
			required = true
			action	 = :store_arg
			arg_type = String
		"--DS-current"
			help	 = "Current Drug-Substructure similarity matrix"
			required = true
			action	 = :store_arg
			arg_type = String
		"--DS-future"
			help	 = "Future Drug-Substructure similarity matrix"
			required = true
			action	 = :store_arg
			arg_type = String
		"--output-file", "-o"
			help	 = "Prediction file name template"
			required = true
			action	 = :store_arg
			arg_type = String
	end

	parsed_args = parse_args(args, configs)

	# Load matrices to memory
	DT₀	   = readdlm(parsed_args["DT-current"])
	DS₀	   = readdlm(parsed_args["DS-current"])
	DTΔ	   = readdlm(parsed_args["DT-future"])
	DSΔ	   = readdlm(parsed_args["DS-future"])

	predict((DT₀,DTΔ), (DS₀, DSΔ), parsed_args["output-file"])
end

main(ARGS)
