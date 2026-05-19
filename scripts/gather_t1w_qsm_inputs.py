#!/usr/bin/env python

from antsnetct import bids_helpers,system_helpers

import ants

import argparse
import json
import logging
import os
import pandas as pd
import sys
import tempfile

logger = logging.getLogger(__name__)

# Helps with CLI help formatting
class RawDefaultsHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass

def gather_inputs():

    parser = argparse.ArgumentParser(formatter_class=RawDefaultsHelpFormatter, add_help = False,
                                     description='''Gather T1w and QSM images for registration.

    The required input file is a list of participants and sessions to process, in CSV format with columns 'participant'
    and 'session'.

    Output is to a BIDS derivative dataset.

    The target T1w to use for registration is based on an FTDC heuristic, if there is more than one T1w image for the session.

    The QSM image is assumed to be named 'Sepia_part-mag.nii.gz' and located in a sub-subject/ses-session/output directory.

    --- Processing steps ---

    1. T1w selection. If there is more than one T1w image for the session, the best one is selected based on an FTDC
    heuristic. If there is only one T1w image, it is used.

    2. QSM selection. The first volume from the session QSM image 'Sepia_part-mag.nii.gz' is is used as the target for
    registration.


    ''')
    required_parser = parser.add_argument_group("Required arguments")
    required_parser.add_argument("--antsnetct-dataset", help="BIDS dataset dir, containing the source T1w images and antsnetct "
                                 "derivatives. The script looks for images matching '_desc-preproc_T1w.nii.gz' as input",
                                 type=str, required=True)
    required_parser.add_argument("--session-list", help="CSV file with participants and sessions to process", type=str, required=True)
    required_parser.add_argument("--qsm-dir", help="Sepia output directory containing the QSM images. This is not a BIDS "
                                 "dataset, though it has a sub-subject/ses-session directory structure", type=str,
                                 required=True)
    required_parser.add_argument("--output-dataset", help="Output BIDS dataset dir", type=str, required=True)

    optional_parser = parser.add_argument_group("General optional arguments")
    optional_parser.add_argument("-h", "--help", action="help", help="show this help message and exit")
    optional_parser.add_argument("--verbose", help="Verbose output from subcommands", action='store_true')

    if len(sys.argv) == 1:
        parser.print_usage()
        print(f"\nRun {os.path.basename(sys.argv[0])} --help for more information")
        sys.exit(1)

    args = parser.parse_args()

    logger.info("Parsed args: " + str(args))

    system_helpers.set_verbose(args.verbose)

    input_dataset = args.antsnetct_dataset
    output_dataset = args.output_dataset

    if (os.path.realpath(input_dataset) == os.path.realpath(output_dataset)):
        raise ValueError('Input and output datasets cannot be the same')

    input_dataset_description = None

    if os.path.exists(os.path.join(input_dataset, 'dataset_description.json')):
        with open(os.path.join(input_dataset, 'dataset_description.json'), 'r') as f:
            input_dataset_description = json.load(f)
    else:
        raise ValueError('Input dataset does not contain a dataset_description.json file')

    logger.info("Input dataset path: " + input_dataset)
    logger.info("Input dataset name: " + input_dataset_description['Name'])

    output_dataset_link_paths = [os.path.abspath(input_dataset)]

    bids_helpers.update_output_dataset(output_dataset, input_dataset_description['Name'] + '_t1w_to_qsm',
                                       output_dataset_link_paths)

    with open(os.path.join(output_dataset, 'dataset_description.json'), 'r') as f:
        output_dataset_description = json.load(f)

    logger.info("Output dataset path: " + output_dataset)
    logger.info("Output dataset name: " + output_dataset_description['Name'])

    bids_t1w_filter = bids_helpers.get_modality_filter_query('t1w')
    bids_t1w_filter['desc'] = 'preproc'

    # Read input from csv
    with open(args.session_list, 'r') as f:
        session_df = pd.read_csv(f, names=['participant', 'session'])

    for participant, session in zip(session_df['participant'], session_df['session']):
        with tempfile.TemporaryDirectory(suffix=f"qsm_t1w_selector.tmpdir") as work_dir:
            logger.info(f"Processing participant {participant}, session {session}")

            if os.path.exists(os.path.join(output_dataset, f"sub-{participant}", f"ses-{session}", 'anat')):
                logger.info(f"Outputs already exist for participant {participant}, session {session}")
                continue

            bids_t1w_filter['session'] = session

            input_t1w_bids = bids_helpers.find_participant_images(input_dataset, participant, work_dir, validate=False,
                                                          **bids_t1w_filter)

            if input_t1w_bids is None or len(input_t1w_bids) == 0:
                logger.warning(f"No T1w images found for participant {participant}.")
                continue

            for t1w_bids in input_t1w_bids:
                logger.info("Found T1w image: " + t1w_bids.get_uri(relative=False))

            selected_t1w_bids = select_best_t1w_image(input_t1w_bids)

            # Multi-echo QSM magnitude image
            qsm_image_path = os.path.join(args.qsm_dir, f"sub-{participant}", f"ses-{session}", "output",
                                          "Sepia_part-mag.nii.gz")

            if not os.path.exists(qsm_image_path):
                logger.warning(f"QSM image not found for participant {participant}, session {session}: {qsm_image_path}")
                continue

            qsm_ref_input = get_qsm_reference_image(args.qsm_dir, participant, session, work_dir)

            # Copy images to output dataset
            qsm_output_rel_path = os.path.join(f"sub-{participant}", f"ses-{session}", "anat",
                                               f"sub-{participant}_ses-{session}_desc-SepiaEcho1_magnitude.nii.gz")

            qsm_ref_bids = bids_helpers.image_to_bids(qsm_ref_input, output_dataset, qsm_output_rel_path,
                                                      metadata={'Sources': [qsm_image_path], 'SkullStripped': False})

            output_t1w_bids = bids_helpers.image_to_bids(selected_t1w_bids.get_path(), output_dataset,
                                                         selected_t1w_bids.get_rel_path(), metadata={
                                                    'Sources': [selected_t1w_bids.get_uri(relative=False)],
                                                    'SkullStripped': False
                                                            })

            # Copy masks
            selected_t1w_mask_bids = selected_t1w_bids.get_derivative_image('_desc-brain_mask.nii.gz')

            output_t1w_mask_bids = bids_helpers.image_to_bids(
                selected_t1w_mask_bids.get_path(), output_dataset,
                output_t1w_bids.get_derivative_rel_path_prefix() + '_desc-antsnetct_mask.nii.gz',
                metadata={'Sources': [selected_t1w_mask_bids.get_uri(relative=False)]}
                )

            qsm_input_mask = get_qsm_mask_image(args.qsm_dir, participant, session)

            output_qsm_mask_bids = bids_helpers.image_to_bids(
                qsm_input_mask, output_dataset,
                os.path.join(f"sub-{participant}", f"ses-{session}", "anat",
                             f"sub-{participant}_ses-{session}_desc-sepia_mask.nii.gz"),
                metadata={'Sources': [qsm_input_mask]}
                )


def get_qsm_reference_image(qsm_dir, participant, session, work_dir):
    """
    Get the reference QSM image for registration.

    The first volume from the QSM image is used as the reference.

    Args:
        qsm_dir (str): Directory containing the QSM images.
        participant (str): Participant ID.
        session (str): Session ID.
        work_dir (str): Working directory for temporary files.

    Returns:
        str: reference QSM brain image.
    """
    if not os.path.exists(qsm_dir):
        raise ValueError(f"QSM image not found: {qsm_dir}")

    qsm_image_path = os.path.join(qsm_dir, f"sub-{participant}", f"ses-{session}", "output", "Sepia_part-mag.nii.gz")

    qsm_image = ants.image_read(qsm_image_path)
    if qsm_image is None:
        raise ValueError(f"Failed to read QSM image: {qsm_image_path}")

    qsm_image_np = qsm_image.numpy()
    # get first volume
    qsm_ref_np = qsm_image_np[:,:,:,0]

    qsm_ref = ants.from_numpy(qsm_ref_np, origin=qsm_image.origin[:3], spacing=qsm_image.spacing[:3],
                             direction=qsm_image.direction[:3,:3])

    qsm_ref_output_fn = os.path.join(work_dir, f"sub-{participant}_ses-{session}_qsm_ref.nii.gz")

    ants.image_write(qsm_ref, qsm_ref_output_fn)

    return qsm_ref_output_fn


def get_qsm_mask_image(qsm_dir, participant, session):
    """
    Get the QSM brain mask image for registration.

    Args:
        qsm_dir (str): Directory containing the QSM images.
        participant (str): Participant ID.
        session (str): Session ID.
        work_dir (str): Working directory for temporary files.

    Returns:
        str: QSM brain mask image.
    """
    if not os.path.exists(qsm_dir):
        raise ValueError(f"QSM image not found: {qsm_dir}")

    qsm_mask_path = os.path.join(qsm_dir, f"sub-{participant}", f"ses-{session}", "output", "Sepia_mask_QSM.nii.gz")

    if not os.path.exists(qsm_mask_path):
        raise ValueError(f"QSM mask image not found: {qsm_mask_path}")

    return qsm_mask_path


def select_best_t1w_image(t1w_bids_list):
    """
    Select the best T1w image from a list of T1w BIDSImage objects, based on an FTDC heuristic.

    If there is only one image in the list, it is returned.

    If there are multiple images, the one with the highest resolution (smallest voxel size) is selected.
    If there is a tie in resolution, the image with the smallest file size is selected.

    Args:
        t1w_bids_list (list): List of BIDSImage objects representing T1w images.

    Returns:
        BIDSImage: The selected best T1w image.
    """
    if len(t1w_bids_list) == 0:
        raise ValueError("The input list of T1w BIDS images is empty.")

    if len(t1w_bids_list) == 1:
        return t1w_bids_list[0]

    # List of t1ws of each type
    abcd = list()
    abcdnd = list()
    mprage1 = list()
    vnavpass = list()
    vnavmoco = list()
    recnorm = list()
    acqsag = list()
    others = list()


    for t1w_bids in t1w_bids_list:
        entities = t1w_bids.get_file_entities()
        if entities.get('acq', None) == 'ABCD800um':
            abcd.append(t1w_bids)
        elif entities.get('acq', None) == 'ABCD800umxND':
            abcdnd.append(t1w_bids)
        elif entities.get('acq', None) == 'MPRAGE1mm':
            mprage1.append(t1w_bids)
        elif entities.get('acq', None) == 'vnavpass':
            vnavpass.append(t1w_bids)
        elif entities.get('acq', None) == 'vnavmoco':
            vnavmoco.append(t1w_bids)
        elif entities.get('rec', None) == 'norm':
            recnorm.append(t1w_bids)
        elif entities.get('acq', None) == 'sag':
            acqsag.append(t1w_bids)
        else:
            others.append(t1w_bids)

    if len(abcd) > 0:
        return get_last_run(abcd)
    if len(abcdnd) > 0:
        return get_last_run(abcdnd)
    if len(mprage1) > 0:
        return get_last_run(mprage1)
    if len(vnavpass) > 0:
        return get_last_run(vnavpass)
    if len(vnavmoco) > 0:
        return get_last_run(vnavmoco)
    if len(recnorm) > 0:
        return get_last_run(recnorm)
    if len(acqsag) > 0:
        return get_last_run(acqsag)
    return get_last_run(others)


def get_last_run(t1w_bids_list):
    """
    Get the T1w image with the highest run number from a list of T1w BIDSImage objects.

    Args:
        t1w_bids_list (list): List of BIDSImage objects representing T1w images.

    Returns:
        BIDSImage: The T1w image with the highest run number.
    """
    if len(t1w_bids_list) == 0:
        raise ValueError("The input list of T1w BIDS images is empty.")
    if len(t1w_bids_list) == 1:
        return t1w_bids_list[0]

    max_run = -1
    selected_t1w = None

    for t1w_bids in t1w_bids_list:
        entities = t1w_bids.get_file_entities()
        run = int(entities.get('run', 1))
        if run > max_run:
            max_run = run
            selected_t1w = t1w_bids

    return selected_t1w

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    gather_inputs()
