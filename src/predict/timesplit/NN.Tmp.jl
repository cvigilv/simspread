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

function predict(DDs::Tuple, DTs::Tuple, fout::String)
	DT₀, DTΔ = DTs
	_  , DDΔ = DDs
	
	@show Nc = size(DDΔ, 1)
	@show Nt = size(DT₀, 2)

    for C in 1:1:Nc
        # Predict targets for compound
		F₁ = useGPU(DDΔ[C,:]) .* useGPU(DT₀)
		R  = mapslices(maximum, F₁; dims = 1)

        # Clean predictions of possible errors produced by the data splitting
        DTΔ_Kₜ = [ sum(col) for col in eachcol(DTΔ) ]

		for (i,k) in enumerate(DTΔ_Kₜ)
            if k == 0
                R[i] = -99
            end
        end

        # Save predictions to file
        open(fout, "a+") do f
            for T in 1:1:Nt
                write(f, "$C, $C, $T, $(R[T]), $(DTΔ[C,T])\n")
            end
        end
    end
end

function main(args)
	configs = ArgParseSettings()

	add_arg_group!(configs, "I/O options:")
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
		"--DD-current"
			help	 = "Current Drug-Drug similarity matrix"
			required = true
			action	 = :store_arg
			arg_type = String
		"--DD-future"
			help	 = "Future Drug-Drug similarity matrix"
			required = true
			action	 = :store_arg
			arg_type = String
        "--output-file", "-o"
			arg_type = String
			action	 = :store_arg
			help	 = "File path for predictions"
			required = true
	end

	parsed_args = parse_args(args, configs)

	# Store arguments to variables
	fout   = parsed_args["output-file"]

    # Load matrices to memory
	DT₀	   = readdlm(parsed_args["DT-current"])
	DD₀	   = readdlm(parsed_args["DD-current"])
	DTΔ	   = readdlm(parsed_args["DT-future"])
	DDΔ	   = readdlm(parsed_args["DD-future"])

	# Predict drug-target interactions
	predict((DD₀,DDΔ), (DT₀, DTΔ), fout)
end

main(ARGS)
