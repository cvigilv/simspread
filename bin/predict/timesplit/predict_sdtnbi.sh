#!/usr/bin/env bash

# Set number of threads to use for matrix multiplication
# OPENBLAS_NUM_THREADS=6

# Useful paths
ROOT=$(git rev-parse --show-toplevel)
DATA="${ROOT}/data/chembl"
SRC="${ROOT}/src/predict/timesplit"

# Parameters
DESCRIPTORS="maccs fp2 ecfp4 fcfp4 kr"

# Time-split validation {{{
for FP in $DESCRIPTORS;
do
	DT_OLD="${DATA}/DT/chembl24_DT.txt"
	DT_NEW="${DATA}/DT/c24_to_c28_DT.txt"
	DS_OLD="${DATA}/DS/chembl24_D.${FP}.txt"
	DS_NEW="${DATA}/DS/c24_to_c28_D.${FP}.txt"
	DT_PRD="${ROOT}/results/timesplit/sdtnbi/sdtnbi.chembl_timesplit.${FP}.out"
	
	julia "${SRC}/SDTNBI.Tmp.jl" \
		--DT-current "$DT_OLD" \
		--DT-future  "$DT_NEW" \
		--DS-current "$DS_OLD" \
		--DS-future  "$DS_NEW" \
		-o "$DT_PRD"
done # }}}

python -Wignore "${ROOT}/src/evaluate/evaluate_simspread.py" \
	"${ROOT}/results/timesplit/sdtnbi/*.out" \
	"${ROOT}/results/timesplit/sdtnbi/sdtnbi.chembl_timesplit"
