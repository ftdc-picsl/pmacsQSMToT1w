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
  $0 -a antsnetct_dataset -i gathered_input_dataset -o output_dataset -m mask_method subj_sess_list.csv
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

    -i gathered_input_dataset : BIDS dataset dir, containing the selected T1w, QSM, and masks for alignment.

    -o output_dataset : Output BIDS dataset dir, where the registration transforms and label images warped
                        to QSM space will be stored.

    -m mask_method :  method for selecting brain masks for T1w images. Options are "synthstrip",
                      "synthstrip_no_csf", or "no_synthstrip". If the latter, masks from antsnetct (hd-bet)
                      for T1w and Sepia (for QSM) will be used.

  Positional args:

    subj_sess_list.csv : CSV file with participants and sessions to process, one per line, no header.


  Output:

  Output is to a BIDS derivative dataset. Transform files and warped images are stored as derivatives of the QSM image.


HELP

}

antsnetct_dataset=""
input_dataset=""
mask_method=""
output_dataset=""

while getopts "a:i:m:o:h" opt; do
  case $opt in
    a) antsnetct_dataset=$OPTARG;;
    i) input_dataset=$OPTARG;;
    m) mask_method=$OPTARG;;
    o) output_dataset=$OPTARG;;
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

nthreads=2

export APPTAINERENV_ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$nthreads
export APPTAINERENV_OMP_NUM_THREADS=$nthreads

for line in $(cat ${imageList}); do
  participant=$(echo ${line} | cut -d ',' -f 1)
  session=$(echo ${line} | cut -d ',' -f 2)

  # Check if output exists under the output bids dataset, and skip if so
  session_outdir="${output_dataset}/sub-${participant}/ses-${session}/anat"

  if [ -d "${session_outdir}" ]; then
    echo "Output directory ${session_outdir} already exists, skipping participant ${participant}, session ${session}"
    continue
  fi

  echo "Submitting registration for participant ${participant}, session ${session}"

  bsub \
    -cwd . \
    -o "${output_dataset}/code/logs/register_t1w_to_qsm_${participant}_${session}_${date}_%J.txt" \
    -n $nthreads \
    -J "regT1wQSM_${participant}_${session}" \
    apptainer exec \
      --containall \
      -B /scratch:/tmp,${antsnetct_dataset},${input_dataset},${output_dataset},${repoDir},${imageList} \
      ${container} \
        ${repoDir}/scripts/register_t1w_to_qsm.py \
        --antsnetct-dataset ${antsnetct_dataset} \
        --input-dataset ${input_dataset} \
        --registration-mask-strategy ${mask_method} \
        --output-dataset ${output_dataset} \
        --participant ${participant} \
        --session ${session} \
        --verbose
  sleep 1

done
