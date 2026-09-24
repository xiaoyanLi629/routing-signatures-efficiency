#!/usr/bin/env python3
"""
HCP Multi-Task fMRI Data Downloader
====================================

Download HCP S1200 task fMRI data (all 7 cognitive tasks) via AWS S3.
Automatically obtains S3 credentials from BALSA (balsa.wustl.edu).

Adapted from Network Stability project's download_hcp.py (WM-only) to
fetch all 7 HCP tasks required by this project's multi-task analysis.

Usage:
    # Download 100 subjects with all 7 tasks (default)
    python tools/download_hcp.py --output /root/autodl-fs/hcp_data

    # Download specific number of subjects
    python tools/download_hcp.py --n-subjects 10 --output /root/autodl-fs/hcp_data

    # Download specific tasks only
    python tools/download_hcp.py --tasks WM MOTOR --output /root/autodl-fs/hcp_data

    # Dry run
    python tools/download_hcp.py --n-subjects 5 --dry-run --output /root/autodl-fs/hcp_data

    # Resume interrupted download (skips existing files)
    python tools/download_hcp.py --output /root/autodl-fs/hcp_data

    # Provide credentials manually (skip BALSA login)
    HCP_AWS_KEY=xxx HCP_AWS_SECRET=yyy python tools/download_hcp.py --output ...
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('hcp_download')

# =============================================================================
# TASK-SPECIFIC FILE LISTS
# =============================================================================

# Files shared across all tasks/runs (same template names)
COMMON_FILES_PER_RUN = [
    'Movement_Regressors.txt',
    'Movement_Regressors_dt.txt',
    'Movement_RelativeRMS.txt',
    'Movement_RelativeRMS_mean.txt',
]

# Task-specific EV timing files (per run/EV directory)
EV_FILES = {
    'WM': [
        '0bk_body.txt', '0bk_faces.txt', '0bk_places.txt', '0bk_tools.txt',
        '2bk_body.txt', '2bk_faces.txt', '2bk_places.txt', '2bk_tools.txt',
        '0bk_cor.txt', '0bk_err.txt', '0bk_nlr.txt',
        '2bk_cor.txt', '2bk_err.txt', '2bk_nlr.txt',
    ],
    'MOTOR': [
        'cue.txt', 'lf.txt', 'lh.txt', 'rf.txt', 'rh.txt', 't.txt',
    ],
    'LANGUAGE': [
        'math.txt', 'story.txt',
        'present_math.txt', 'present_story.txt',
        'question_math.txt', 'question_story.txt',
        'response_math.txt', 'response_story.txt',
    ],
    'SOCIAL': [
        'mental.txt', 'rnd.txt',
        'mental_resp.txt', 'other_resp.txt',
    ],
    'RELATIONAL': [
        'match.txt', 'relation.txt',
        'error.txt',
    ],
    'EMOTION': [
        'fear.txt', 'neut.txt',
    ],
    'GAMBLING': [
        'win.txt', 'loss.txt', 'neut.txt',
        'win_event.txt', 'loss_event.txt', 'neut_event.txt',
    ],
}

# Task-specific behavioral stats files (optional; may not exist for all tasks)
STATS_FILES = {
    'WM':         ['WM_Stats.csv'],
    'MOTOR':      [],
    'LANGUAGE':   ['Language_Stats.csv'],
    'SOCIAL':     ['Social_Stats.csv'],
    'RELATIONAL': ['Relational_Stats.csv'],
    'EMOTION':    ['Emotion_Stats.csv'],
    'GAMBLING':   ['Gambling_Stats.csv'],
}

ALL_TASKS = ['WM', 'MOTOR', 'LANGUAGE', 'SOCIAL', 'RELATIONAL', 'EMOTION', 'GAMBLING']

# 200 subjects with complete task fMRI data (from Network Stability project
# reference list — verified to have WM; assumed to have all 7 tasks since HCP
# inclusion criteria require all task completions).
REFERENCE_SUBJECTS = [
    '100206', '100307', '100408', '100610', '101006', '101107', '101309',
    '101915', '102008', '102311', '102513', '102614', '102715', '102816',
    '103111', '103212', '103414', '103515', '104416', '104820',
    '105014', '105115', '105216', '105923', '106016', '106319', '106521',
    '107018', '107321', '107422', '108121', '108222', '108323', '108525',
    '108828', '109123', '109325', '110007', '110411', '111312', '111413',
    '111514', '111716', '112112', '112314', '112516', '112920', '113215',
    '113619', '113922', '114217', '114318', '114419', '114621', '114823',
    '114924', '115017', '115320', '115825', '116524', '116726', '117122',
    '117324', '117930', '118023', '118124', '118225', '118528', '118730',
    '118831', '118932', '119126', '120111', '120212', '120515', '120717',
    '121416', '121618', '121921', '122317', '122620', '122822', '123117',
    '123420', '123521', '123824', '123925', '124220', '124422', '124624',
    '124826', '125525', '126325', '126628', '127226', '127630', '127933',
    '128026', '128127', '128329', '128632', '128935', '129028', '129129',
    '129331', '129634', '130013', '130316', '130417', '130518', '130619',
    '130720', '130821', '130922', '131217', '131419', '131722', '131823',
    '131924', '132017', '132118', '133019', '133625', '133827', '133928',
    '134021', '134223', '134324', '134425', '134728', '134829', '135225',
    '135528', '135730', '135932', '136126', '136227', '136631', '136732',
    '136833', '137027', '137128', '137229', '137431', '137532', '137633',
    '137936', '138231', '138332', '138534', '138837', '139233', '139435',
    '139637', '139839', '140117', '140420', '140824', '140925', '141119',
    '141422', '141826', '142828', '143325', '143426', '143830', '144226',
    '144428', '144731', '144832', '145127', '145834', '146129', '146331',
    '146432', '146533', '146634', '146735', '146937', '147030', '147737',
    '148032', '148133', '148335', '148436', '148840', '148941', '149236',
    '149337', '149539', '149741', '149842', '150423', '150524', '150625',
    '150726', '150928', '151223', '151425', '151526', '151627', '151728',
    '151829', '152225', '152427', '152831', '153025', '153126', '153227',
]

BUCKET = 'hcp-openaccess'
PREFIX = 'HCP_1200'

# Approximate per-subject download size (all 7 tasks, both LR+RL)
# Each Atlas dtseries ~150 MB; motion/EVs negligible.
# 7 tasks * 2 runs * ~150 MB + overhead ≈ ~2.1 GB per subject
SIZE_PER_SUBJECT_GB = 2.1


# =============================================================================
# CREDENTIALS
# =============================================================================

def get_credentials_from_balsa():
    """Obtain HCP S3 credentials from BALSA platform.

    Expects credentials via environment variables:
        BALSA_USERNAME, BALSA_PASSWORD
    Register at https://balsa.wustl.edu/ and accept the HCP Open Access
    Data Use Terms before use.
    """
    import requests
    username = os.environ.get('BALSA_USERNAME')
    password = os.environ.get('BALSA_PASSWORD')
    if not username or not password:
        log.error('Set BALSA_USERNAME and BALSA_PASSWORD environment '
                  'variables, or pass HCP_AWS_KEY/HCP_AWS_SECRET directly.')
        sys.exit(1)
    log.info('Obtaining S3 credentials from BALSA...')
    session = requests.Session()
    session.get('https://balsa.wustl.edu/login/auth', timeout=15)
    session.post('https://balsa.wustl.edu/j_spring_security_check',
                 data={'j_username': username, 'j_password': password},
                 timeout=15)
    r = session.get('https://balsa.wustl.edu/project/aws?project=HCP_YA', timeout=30)
    if r.status_code != 200:
        log.error(f'BALSA returned status {r.status_code}')
        sys.exit(1)
    data = r.text.strip()
    if data.startswith('"'):
        data = json.loads(data)
    creds = json.loads(data) if isinstance(data, str) else data
    log.info(f'Got S3 credentials (Access Key: {creds["accessKeyId"][:12]}...)')
    log.info('Waiting 20s for AWS key propagation...')
    time.sleep(20)
    return creds['accessKeyId'], creds['secretAccessKey']


def get_credentials():
    """Get S3 credentials from env vars or BALSA."""
    key = os.environ.get('HCP_AWS_KEY')
    secret = os.environ.get('HCP_AWS_SECRET')
    if key and secret:
        log.info('Using credentials from environment variables')
        return key, secret
    return get_credentials_from_balsa()


# =============================================================================
# DOWNLOADERS
# =============================================================================

def _download_one(s3, s3_key, local_path, max_retries=3):
    """Download a single file with retries. Returns True on success."""
    from boto3.s3.transfer import TransferConfig
    transfer_config = TransferConfig(max_concurrency=2)
    for attempt in range(max_retries):
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, s3_key, str(local_path), Config=transfer_config)
            return True
        except Exception as e:
            if local_path.exists():
                local_path.unlink()
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                log.debug(f'  Retry {attempt+1} in {wait}s: {s3_key.split("/")[-1]}')
                time.sleep(wait)
            else:
                log.warning(f'  FAIL {s3_key}: {e}')
                return False


def download_subject_task(s3, subject, task, output_dir, dry_run=False):
    """Download files for one subject/task pair (both LR and RL runs).

    Returns (n_downloaded, n_skipped, n_failed).
    """
    downloaded, skipped, failed = 0, 0, 0

    for run in ['LR', 'RL']:
        base_s3 = f'{PREFIX}/{subject}/MNINonLinear/Results/tfMRI_{task}_{run}'
        local_base = (Path(output_dir) / subject / 'MNINonLinear' / 'Results'
                      / f'tfMRI_{task}_{run}')

        # Main dtseries + motion
        main_files = [
            f'tfMRI_{task}_{run}_Atlas_MSMAll.dtseries.nii',
        ] + COMMON_FILES_PER_RUN

        for fname in main_files:
            s3_key = f'{base_s3}/{fname}'
            local_path = local_base / fname

            if local_path.exists() and local_path.stat().st_size > 0:
                skipped += 1
                continue
            if dry_run:
                log.info(f'  [DRY] {subject}/{task}/{run}/{fname}')
                continue
            if _download_one(s3, s3_key, local_path):
                downloaded += 1
            else:
                failed += 1

        # EV timing files
        for ev in EV_FILES.get(task, []):
            s3_key = f'{base_s3}/EVs/{ev}'
            local_path = local_base / 'EVs' / ev

            if local_path.exists() and local_path.stat().st_size > 0:
                skipped += 1
                continue
            if dry_run:
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if _download_one(s3, s3_key, local_path, max_retries=2):
                downloaded += 1
            # EV file misses are common (not all subjects have all EVs); don't
            # increment failed counter for EVs.

        # Behavioral stats (HCP stores these under EVs/ subdirectory)
        for stats_file in STATS_FILES.get(task, []):
            s3_key = f'{base_s3}/EVs/{stats_file}'
            local_path = local_base / 'EVs' / stats_file
            if local_path.exists() and local_path.stat().st_size > 0:
                skipped += 1
                continue
            if dry_run:
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if _download_one(s3, s3_key, local_path, max_retries=2):
                downloaded += 1

    return downloaded, skipped, failed


def download_subject(s3, subject, tasks, output_dir, dry_run=False):
    """Download all requested tasks for one subject.

    Returns (n_downloaded, n_skipped, n_failed).
    """
    total_dl, total_skip, total_fail = 0, 0, 0
    for task in tasks:
        dl, sk, fl = download_subject_task(s3, subject, task, output_dir, dry_run)
        total_dl += dl
        total_skip += sk
        total_fail += fl
    return total_dl, total_skip, total_fail


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Download HCP multi-task fMRI data')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--n-subjects', type=int, default=100,
                        help='Number of subjects to download (default: 100)')
    parser.add_argument('--subjects', nargs='+',
                        help='Specific subject IDs (overrides --n-subjects)')
    parser.add_argument('--tasks', nargs='+', default=ALL_TASKS,
                        choices=ALL_TASKS,
                        help=f'Tasks to download (default: all 7 = {ALL_TASKS})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show plan without downloading')
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel download workers (default: 4)')
    parser.add_argument('--skip-existing-subjects', action='store_true',
                        help='Skip subjects that already have a non-empty directory')
    args = parser.parse_args()

    subjects = (args.subjects if args.subjects
                else REFERENCE_SUBJECTS[:args.n_subjects])

    if args.skip_existing_subjects:
        filtered = []
        for s in subjects:
            subj_dir = Path(args.output) / s / 'MNINonLinear' / 'Results'
            # Consider "has data" if any requested task directory exists
            has_any = any((subj_dir / f'tfMRI_{t}_LR').exists() for t in args.tasks)
            if not has_any:
                filtered.append(s)
        skipped = len(subjects) - len(filtered)
        log.info(f'Skipping {skipped} subjects with existing data')
        subjects = filtered

    log.info(f'HCP Multi-Task Data Downloader')
    log.info(f'  Subjects: {len(subjects)}')
    log.info(f'  Tasks:    {args.tasks}')
    log.info(f'  Output:   {args.output}')
    log.info(f'  Workers:  {args.workers}')
    task_factor = len(args.tasks) / len(ALL_TASKS)
    log.info(f'  Est. size: ~{len(subjects) * SIZE_PER_SUBJECT_GB * task_factor:.1f} GB')

    if args.dry_run:
        log.info('  Mode: DRY RUN')

    if not subjects:
        log.info('Nothing to download. Exiting.')
        return

    key, secret = get_credentials()

    import boto3
    from botocore.config import Config as BotoConfig

    total_dl, total_skip, total_fail = 0, 0, 0

    def process_subject(subj_idx_and_id):
        idx, subj = subj_idx_and_id
        s3 = boto3.client('s3',
                          aws_access_key_id=key,
                          aws_secret_access_key=secret,
                          region_name='us-east-1',
                          config=BotoConfig(
                              max_pool_connections=10,
                              retries={'max_attempts': 3, 'mode': 'adaptive'},
                              connect_timeout=30,
                              read_timeout=120,
                          ))
        dl, sk, fl = download_subject(s3, subj, args.tasks, args.output, args.dry_run)
        return idx, subj, dl, sk, fl

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_subject, (i, s)): s
                   for i, s in enumerate(subjects)}

        for future in as_completed(futures):
            idx, subj, dl, sk, fl = future.result()
            total_dl += dl
            total_skip += sk
            total_fail += fl
            status = f'dl={dl} skip={sk}'
            if fl > 0:
                status += f' FAIL={fl}'
            log.info(f'[{idx+1}/{len(subjects)}] {subj}: {status}')

    log.info(f'\nDone! Downloaded={total_dl}, Skipped={total_skip}, Failed={total_fail}')
    log.info(f'Data location: {args.output}')


if __name__ == '__main__':
    main()
