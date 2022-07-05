#!/usr/bin/env bash

# Set number of threads to use for matrix multiplication
# OPENBLAS_NUM_THREADS=6

# Useful paths
ROOT=$(git rev-parse --show-toplevel)
SRC="${ROOT}/src/predict/10fold"

# Parameters
DESCRIPTORS="maccs fp2 ecfp4 fcfp4 kr"

# Yamanishi, et al (2008) {{{
YAMANISHI="${ROOT}/data/yamanishi2008"
YAMANISHI_DATASETS="nr ic gpcr e"
for DATASET in $YAMANISHI_DATASETS;
do
	for FP in $DESCRIPTORS;
	do
		DT="${YAMANISHI}/DT/${DATASET}_DT.txt"
		DS="${YAMANISHI}/DS/${DATASET}_DS.${FP}.txt"
		DT_PRD="${ROOT}/results/10fold/sdtnbi/${DATASET}.sdtnbi_10foldNN.${FP}.out"

		julia "${SRC}/SDTNBI.kFold.jl" \
			--DT "$DT" \
			--DS "$DS" \
            -n 10 \
            -k 10 \
            -s 0 \
			-o "$DT_PRD"
	done
done
# }}}
# Wu, et al (2017) {{{
WU="${ROOT}/data/global2017"
WU_DATASETS="global"
for DATASET in $WU_DATASETS;
do
	for FP in $DESCRIPTORS;
	do
		DT="${WU}/DT/${DATASET}_DT.txt"
		DS="${WU}/DS/${DATASET}_DS.${FP}.txt"
		DT_PRD="${ROOT}/results/10fold/sdtnbi/${DATASET}.sdtnbi_10foldNN.${FP}.out"

		julia "${SRC}/SDTNBI.kFold.jl" \
			--DT "$DT" \
			--DS "$DS" \
            -n 10 \
            -k 10 \
            -s 0 \
			-o "$DT_PRD"
	done
done
# }}}

python -Wignore "${ROOT}/src/evaluate/evaluate_simspread.py" \
	"${ROOT}/results/10fold/sdtnbi/global*.out" \
	"${ROOT}/results/10fold/sdtnbi/wu.sdtnbi_10fold"
