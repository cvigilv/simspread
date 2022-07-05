#!/usr/bin/env bash

# Set number of threads to use for matrix multiplication
# OPENBLAS_NUM_THREADS=6

# Useful paths
ROOT=$(git rev-parse --show-toplevel)
SRC="${ROOT}/src/predict/loo"

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
		DD="${YAMANISHI}/DS/${DATASET}_DS.${FP}.txt"
		DT_PRD="${ROOT}/results/loo/sdtnbi/${DATASET}.sdtnbi_loo.${FP}.out"

		julia "${SRC}/SDTNBI.LOOCV.jl" \
			--DT "$DT" \
			--DS "$DD" \
			-o "$DT_PRD" &
	done
done # }}}
# Wu, et al (2017) {{{
WU="${ROOT}/data/wu2017"
WU_DATASETS="global"
for DATASET in $WU_DATASETS;
do
	for FP in $DESCRIPTORS;
	do
		DT="${WU}/DT/${DATASET}_DT.txt"
		DD="${WU}/DS/${DATASET}_DS.${FP}.txt"
		DT_PRD="${ROOT}/results/loo/sdtnbi/${DATASET}.sdtnbi_loo.${FP}.out"

		julia "${SRC}/SDTNBI.LOOCV.jl" \
			--DT "$DT" \
			--DS "$DD" \
			-o "$DT_PRD" &
	done
done # }}}

python -Wignore "${ROOT}/src/evaluate/evaluate_simspread.py" \
	"${ROOT}/results/loo/sdtnbi/*.out" \
	"${ROOT}/results/loo/sdtnbi/yamanishi+wu.sdtnbi_loo"
