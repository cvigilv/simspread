#!/usr/bin/env bash

# Set number of threads to use for matrix multiplication
# OPENBLAS_NUM_THREADS=6

# Useful paths
ROOT=$(git rev-parse --show-toplevel)
DATA="${ROOT}/data/chembl"
SRC="${ROOT}/src/predict/timesplit"

# Parameters
DESCRIPTORS="maccs fp2 ecfp4 fcfp4 kr mold2"

# Time-split validation {{{
for FP in $DESCRIPTORS;
do
	DT_OLD="${DATA}/DT/chembl24_DT.txt"
	DT_NEW="${DATA}/DT/c24_to_c28_DT.txt"
	DD_OLD="${DATA}/DD/chembl24_DD.${FP}_tanimoto.txt"
	DD_NEW="${DATA}/DD/c24_to_c28_DD.${FP}_tanimoto.txt"
	DT_PRD="${ROOT}/results/timesplit/simspread/simspread.chembl_timesplit.${FP}"

	julia "${SRC}/SimSpread.Tmp.jl" \
		--DT-current "$DT_OLD" \
		--DT-future  "$DT_NEW" \
		--DD-current "$DD_OLD" \
		--DD-future  "$DD_NEW" \
		-o "$DT_PRD"

	DT_PRD="${ROOT}/results/timesplit/simspread/wsimspread.chembl_timesplit.${FP}"
	julia "${SRC}/SimSpread.Tmp.jl" \
		--DT-current "$DT_OLD" \
		--DT-future  "$DT_NEW" \
		--DD-current "$DD_OLD" \
		--DD-future  "$DD_NEW" \
		--weighted \
		-o "$DT_PRD"
done # }}}

python -Wignore "${ROOT}/src/evaluate/evaluate_simspread.py" \
	"${ROOT}/results/timesplit/simspread/*.out" \
	"${ROOT}/results/timesplit/simspread/simspread.chembl_timesplit"
