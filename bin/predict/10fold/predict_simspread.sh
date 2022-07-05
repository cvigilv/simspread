#!/usr/bin/env bash

# Set number of threads to use for matrix multiplication in CPU
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
	# SimSpread with binary weighting
	for FP in $DESCRIPTORS;
	do
		DT="${YAMANISHI}/DT/${DATASET}_DT.txt"
		DD="${YAMANISHI}/DD/${DATASET}_DD.${FP}_tanimoto.txt"
		DT_PRD="${ROOT}/results/10fold/simspread/${DATASET}.simspread_10foldNN.${FP}"

		julia  "${SRC}/SimSpread.kFold.jl" \
			--DT "$DT" \
			--DD "$DD" \
			-n 10 \
			-k 10 \
			-s 0 \
			-o "$DT_PRD"
	done
	wait

	# SimSpread with chemical similarity weighting
	for FP in $DESCRIPTORS;
	do
		DT="${YAMANISHI}/DT/${DATASET}_DT.txt"
		DD="${YAMANISHI}/DD/${DATASET}_DD.${FP}_tanimoto.txt"
		DT_PRD="${ROOT}/results/10fold/simspread/${DATASET}.wsimspread_10foldNN.${FP}"

		julia  "${SRC}/SimSpread.kFold.jl" \
			--DT "$DT" \
			--DD "$DD" \
			-w \
			-n 10 \
			-k 10 \
			-s 0 \
			-o "$DT_PRD"
	done
	wait
done
# }}}
# Wu, et al (2016) {{{
WU="${ROOT}/data/wu2017"
WU_DATASETS="global"
for DATASET in $WU_DATASETS;
do
	# SimSpread with binary weighting
	for FP in $DESCRIPTORS;
	do
		DT="${WU}/DT/${DATASET}_DT.txt"
		DD="${WU}/DD/${DATASET}_DD.${FP}_tanimoto.txt"
		DT_PRD="${ROOT}/results/10fold/simspread/${DATASET}.simspread_10foldNN.${FP}"

		julia  "${SRC}/SimSpread.kFold.jl" \
			--DT "$DT" \
			--DD "$DD" \
			-n 10 \
			-k 10 \
			-s 0 \
			-o "$DT_PRD"
	done
	wait
	
	# SimSpread with chemical similarity weighting
	for FP in $DESCRIPTORS;
	do
		DT="${WU}/DT/${DATASET}_DT.txt"
		DD="${WU}/DD/${DATASET}_DD.${FP}_tanimoto.txt"
		DT_PRD="${ROOT}/results/10fold/simspread/${DATASET}.wsimspread_10foldNN.${FP}"

		julia  "${SRC}/SimSpread.kFold.jl" \
			--DT "$DT" \
			--DD "$DD" \
			-w \
			-n 10 \
			-k 10 \
			-s 0 \
			-o "$DT_PRD"
	done
	wait
done
# }}}

python -Wignore "${ROOT}/src/evaluate/evaluate_simspread.py" \
	"${ROOT}/results/10fold/simspread/*.out" \
	"${ROOT}/results/10fold/simspread/yamanishi.simspread_10fold"
