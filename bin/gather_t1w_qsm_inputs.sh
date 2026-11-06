#!/bin/bash

module load apptainer/1.4.1

scriptPath=$(readlink -f "$0")
scriptDir=$(dirname "${scriptPath}")
# Repo base dir under which we find bin/ and containers/
repoDir=${scriptDir%/bin}

inputBIDS=""

container="${repoDir}/containers/antsnetct-0.6.2.sif"

function usage() {
  echo "Usage:
  $0 -a antsnetct_dataset -o output_dataset -q qsm_dir subj_sess_list.csv
  "
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

function help() {
cat << HELP
  `usage`

  Wrapper script to organize T1w and QSM data for registration.

  Required args:

    -a antsnetct_dataset : BIDS dataset dir, containing the source T1w images and antsnetct derivatives.
       The script looks for images matching '_desc-preproc_T1w.nii.gz' as input

    -o output_dataset : Output BIDS dataset dir, where the gathered images will be stored

    -q qsm_dir : Sepia output directory containing the QSM images. This is not a BIDS
       dataset, the script looks for files in 'qsm_dir/sub-<participant>/ses-<session>/output'



  Positional args:

    subj_sess_list.csv : CSV file with participants and sessions to process, one per line, no header.


  Output:

  Output is to a BIDS derivative dataset. The following images are gathered for each session:

    - T1w image from antsnetct, stored with suffix '_desc-antsnetct_T1w.nii.gz'

    - Brain mask from antsnetct, stored with suffix '_desc-antsnetct_mask.nii.gz'

    - QSM magnitude image from sepia, stored with suffix '_desc-SepiaEcho1_magnitude.nii.gz'

    - Brain mask from sepia processing, stored with suffix '_desc-sepia_mask.nii.gz'

  The choice of T1w image to use is based on an FTDC heuristic prioritizing images with higher resolution
  and fewer artifacts.


HELP

}

antsnetct_dataset=""
output_dataset=""
qsm_dir=""

while getopts "a:o:q:h" opt; do
  case $opt in
    a) antsnetct_dataset=$OPTARG;;
    o) output_dataset=$OPTARG;;
    q) qsm_dir=$OPTARG;;
    h) help; exit 1;;
    \?) echo "Unknown option $OPTARG"; exit 2;;
    :) echo "Option $OPTARG requires an argument"; exit 2;;
  esac
done

shift $((OPTIND - 1))

imageList=$(readlink -f "$1")

date=`date +%Y%m%d`

mkdir -p ${output_dataset}/code/logs

export APPTAINERENV_TMPDIR="/tmp"

bsub -cwd . -o "${output_dataset}/code/logs/gather_t1w_qsm_inputs_${date}_%J.txt" \
    apptainer exec \
      --containall \
      -B /scratch:/tmp,${antsnetct_dataset},${output_dataset},${qsm_dir},${repoDir},${imageList} \
      ${container} \
        ${repoDir}/scripts/gather_t1w_qsm_inputs.py \
          --antsnetct-dataset ${antsnetct_dataset} \
          --session-list ${imageList} \
          --qsm-dir ${qsm_dir} \
          --output-dataset ${output_dataset}