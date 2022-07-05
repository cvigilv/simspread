#!/usr/bin/env bash

# Set number of threads to use for matrix multiplication
# OPENBLAS_NUM_THREADS=6

# Useful paths
ROOT=$(git rev-parse --show-toplevel)
SRC="${ROOT}/src/predict/10fold"

# Parameters
DESCRIPTORS="maccs fp2 ecfp4 fcfp4 kr mold2"

# Yamanishi, et al (2008) {{{
YAMANISHI="${ROOT}/data/yamanishi2008"
YAMANISHI_DATASETS="nr ic gpcr e"
for DATASET in $YAMANISHI_DATASETS;
do
	for FP in $DESCRIPTORS;
	do
		DT="${YAMANISHI}/DT/${DATASET}_DT.txt"
		DD="${YAMANISHI}/DD/${DATASET}_DD.${FP}_tanimoto.txt"
		DT_PRD="${ROOT}/results/10fold/nn/${DATASET}.nn_10foldNN.${FP}.out"

		julia  "${SRC}/NN.kFold.jl" \
			--DT "$DT" \
			--DD "$DD" \
			-o "$DT_PRD"
	done

done
# }}}
# Wu, et al (2017) {{{
WU="${ROOT}/data/wu2017"
WU_DATASETS="global"
for DATASET in $WU_DATASETS;
do
	for FP in $DESCRIPTORS;
	do
		DT="${WU}/DT/${DATASET}_DT.txt"
		DD="${WU}/DD/${DATASET}_DD.${FP}_tanimoto.txt"
		DT_PRD="${ROOT}/results/10fold/nn/${DATASET}.nn_10foldNN.${FP}.out"

		julia  "${SRC}/NN.kFold.jl" \
			--DT "$DT" \
			--DD "$DD" \
			-o "$DT_PRD"
	done

done
# }}}

python -Wignore "${ROOT}/src/evaluate/evaluate_simspread.py" \
	"${ROOT}/results/10fold/nn/*.out" \
	"${ROOT}/results/10fold/nn/yamanishi+wu.nn_loo"
