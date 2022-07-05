#!/usr/bin/env bash

# Set number of threads to use for matrix multiplication
# OPENBLAS_NUM_THREADS=6

# Useful paths
ROOT=$(git rev-parse --show-toplevel)
SRC="${ROOT}/src/predict/loo"

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
		DT_PRD="${ROOT}/results/loo/simspread/${DATASET}.simspread_loo.${FP}"

		julia  "${SRC}/SimSpread.LOOCV.jl" \
			--DT "$DT" \
			--DS "$DD" \
			-o "$DT_PRD"
	done
	
    # SimSpread with chemical similarity weighting
	for FP in $DESCRIPTORS;
	do
		DT="${YAMANISHI}/DT/${DATASET}_DT.txt"
		DD="${YAMANISHI}/DD/${DATASET}_DD.${FP}_tanimoto.txt"
        DT_PRD="${ROOT}/results/loo/simspread/${DATASET}.wsimspread_loo.${FP}_tanimoto"

		julia  "${SRC}/SimSpread.LOOCV.jl" \
			--DT "$DT" \
			--DS "$DD" \
            --weighted \
			-o "$DT_PRD"
	done
done # }}}
# Wu, et al (2017) {{{
WU="${ROOT}/data/wu2017"
WU_DATASETS="global"
for DATASET in $WU_DATASETS;
do
	# SimSpread with binary weighting
	for FP in $DESCRIPTORS;
	do
		DT="${WU}/DT/${DATASET}_DT.txt"
		DD="${WU}/DD/${DATASET}_DD.${FP}.txt"

		DT_PRD="${ROOT}/results/loo/simspread/${DATASET}.simspread_loo.${FP}"

		julia  "${SRC}/SimSpread.LOOCV.jl" \
			--DT "$DT" \
			--DS "$DD" \
			-o "$DT_PRD"
	done

	for FP in $DESCRIPTORS;
	do
        # SimSpread with chemical similarity weighting
        DT_PRD="${ROOT}/results/loo/simspread/${DATASET}.simspread_loo.${FP}"

		julia  "${SRC}/SimSpread.LOOCV.jl" \
			--DT "$DT" \
			--DS "$DD" \
            --weighted \
			-o "$DT_PRD"
	done
done # }}}

python -Wignore "${ROOT}/src/evaluate/evaluate_simspread.py" \
	"${ROOT}/results/loo/simspread/*.out" \
	"${ROOT}/results/loo/simspread/yamanishi+wu.nn_loo"
