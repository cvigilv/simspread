#!/usr/bin/env bash

# Set number of threads to use for matrix multiplication
# OPENBLAS_NUM_THREADS=6

# Useful paths
ROOT=$(git rev-parse --show-toplevel)
SRC="${ROOT}/src/predict/10fold"

# Yamanishi, et al (2008) {{{
YAMANISHI="${ROOT}/data/yamanishi2008"
YAMANISHI_DATASETS="nr ic gpcr e"
for DATASET in $YAMANISHI_DATASETS;
do
	DT="${YAMANISHI}/DT/${DATASET}_DT.txt"
	DT_PRD="${ROOT}/results/10fold/nbi/${DATASET}.nbi_10foldNN.nan.out"

	julia  "${SRC}/NBI.kFold.jl" \
		--DT "$DT" \
		-o "$DT_PRD"
done
# }}}
# Wu, et al (2016) {{{
WU="${ROOT}/data/wu2017"
WU_DATASETS="global"
	for DATASET in $WU_DATASETS;
	do
		DT="${WU}/DT/${DATASET}_DT.txt"
		DT_PRD="${ROOT}/results/10fold/nbi/${DATASET}.nbi_10foldNN.nan.out"

		julia "${SRC}/NBI.kFold.jl" \
			--DT "$DT" \
			-o "$DT_PRD"
done
# }}}

python -Wignore "${ROOT}/src/evaluate/evaluate_simspread.py" \
	"${ROOT}/results/10fold/nbi/*.out" \
	"${ROOT}/results/10fold/nbi/yamanishi+wu.nbi_10fold"
