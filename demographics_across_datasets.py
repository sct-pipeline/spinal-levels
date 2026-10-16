"""
TL;DR: Summarize sex and age of the subjects used in the spinal levels analyses, per dataset and pooled across datasets.

Datasets (subjects listed in <dataset>/subjects-to-analyse.yml, except whole-spine, see below) and the source of sex
and age:
    - spine-generic (cervical):  participants.tsv of data-multi-subject
    - whole-spine (cervical):    participants.tsv of whole-spine (ds005616)
    - Balgrist (lumbar):         spreadsheet with the spinal level heights (sex only, age not available yet, see
                                 BALGRIST_AGE)
    - EPFL (lumbar):             participants.tsv of the EPFL lumbar dataset
    - Fudan (lumbar):            L1-L5_lengths.xlsx

Whole-spine: only the subjects with the PMJ-to-conus distance (pmj_to_conus_and_cervical_distance_whole-spine.csv) are
used, i.e., a subset of whole-spine-ds005616/subjects-to-analyse.yml.

Output (saved to the -o folder):
    - demographics_per_subject.csv: dataset, subject, sex and age of all subjects
    - demographics_across_datasets.csv: per dataset and pooled: number of subjects, number (%) of females and males
      (and unknown sex), age mean ± SD, median and range

Example:
    python demographics_across_datasets.py

Author: Jan Valosek
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.expanduser('~/data/spinal-levels')
SPINE_GENERIC_TSV = os.path.expanduser('~/data/data.neuro.polymtl.ca/data-multi-subject/participants.tsv')
WHOLE_SPINE_TSV = os.path.expanduser('~/data/data.neuro.polymtl.ca/whole-spine/participants.tsv')
WHOLE_SPINE_CSV = os.path.join(DATA_DIR, 'whole-spine_pmj_to_conus_distance',
                               'pmj_to_conus_and_cervical_distance_whole-spine.csv')
BALGRIST_XLSX = os.path.join(DATA_DIR, 'lumbar_rootlets_Balgrist', 'bids',
                             'spinal_level_heights_top_L1-bottom_S2_with_chart.xlsx')
EPFL_TSV = os.path.join(DATA_DIR, 'lumbar_mri_Sergio', 'participants.tsv')
FUDAN_XLSX = os.path.join(DATA_DIR, 'lumbar_fudan', 'L1-L5_lengths.xlsx')
OUTPUT_DIR = os.path.join(DATA_DIR, 'figures_across_datasets')

# Dataset name: folder (in this repo) with the subjects-to-analyse.yml file
DATASET_YML = {
    'spine-generic': 'spine-generic-multi-subject',
    'whole-spine': 'whole-spine-ds005616',
    'Balgrist': 'lumbar-balgrist',
    'EPFL': 'lumbar-epfl',
    'Fudan': 'lumbar-fudan',
}

# TODO: age of the Balgrist subjects is not in the spreadsheet; fill in once available
BALGRIST_AGE = {
    'sub-ltr02': None,
    'sub-ltr03': None,
    'sub-ltr04': None,
    'sub-ltr05': None,
    'sub-ltr06': None,
    'sub-ltr07': None,
    'sub-ltr08': None,
    'sub-ltr09': None,
    'sub-ltr11': None,
    'sub-ltr13': None,
    'sub-ltr14': None,
    'sub-ltr15': None,
    'sub-ltr17': None,
    'sub-ltr19': None,
    'sub-ltr20': None,
    'sub-ltr23': None,
    'sub-ltr24': None,
}


def read_subjects(dataset):
    """
    Subjects used in the analysis, from <dataset folder>/subjects-to-analyse.yml; for whole-spine, only the subjects
    with the PMJ-to-conus distance.
    """
    if dataset == 'whole-spine':
        return pd.read_csv(WHOLE_SPINE_CSV, sep=';')['subject'].tolist()
    with open(os.path.join(REPO_DIR, DATASET_YML[dataset], 'subjects-to-analyse.yml')) as f:
        return yaml.safe_load(f)


def load_participants_tsv(fname):
    """participant_id, sex and age from a BIDS participants.tsv file."""
    df = pd.read_csv(fname, sep='\t', na_values='n/a')
    return df.rename(columns={'participant_id': 'subject'})[['subject', 'sex', 'age']]


def load_balgrist(fname):
    """Sex from the Balgrist spreadsheet (summary rows below the subjects are dropped); age from BALGRIST_AGE."""
    df = pd.read_excel(fname, sheet_name='Spinal level heights (mm)')
    df = df.rename(columns={'Participants': 'subject'})
    df = df[df['subject'].astype(str).str.startswith('sub-')]
    df['age'] = df['subject'].map(BALGRIST_AGE)
    return df[['subject', 'sex', 'age']]


def load_fudan(fname):
    """Sex and age from the Fudan spreadsheet."""
    return pd.read_excel(fname)[['subject', 'sex', 'age']]


def load_dataset(dataset, df_participants):
    """
    Keep only the subjects listed in the dataset's subjects-to-analyse.yml.
    :return: dataframe with 'dataset', 'subject', 'sex' and 'age' columns
    """
    subjects = read_subjects(dataset)
    missing = sorted(set(subjects) - set(df_participants['subject']))
    if missing:
        print(f'WARNING: {dataset}: {len(missing)} subjects without sex and age: {missing}')
    df = pd.DataFrame({'subject': subjects}).merge(df_participants, on='subject', how='left')
    df['sex'] = df['sex'].str.upper()
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df.insert(0, 'dataset', dataset)
    return df


def summarize(df, name):
    """Number of subjects, sex distribution and age statistics of one group of subjects."""
    n = len(df)
    n_female, n_male = (df['sex'] == 'F').sum(), (df['sex'] == 'M').sum()
    age = df['age'].dropna()
    row = {
        'dataset': name,
        'n': n,
        'n_female': n_female,
        'n_male': n_male,
        'n_sex_unknown': n - n_female - n_male,
        'female_%': round(100 * n_female / n, 1),
        'n_with_age': len(age),
        'age_mean': round(age.mean(), 1) if len(age) else np.nan,
        'age_sd': round(age.std(), 1) if len(age) else np.nan,
        'age_median': age.median() if len(age) else np.nan,
        'age_min': age.min() if len(age) else np.nan,
        'age_max': age.max() if len(age) else np.nan,
    }
    return row


def format_row(row):
    """One line of the summary: n, F/M, age mean ± SD [range]."""
    sex = f"{row['n_female']} F / {row['n_male']} M ({row['female_%']:.0f}% F)"
    if row['n_sex_unknown']:
        sex += f", {row['n_sex_unknown']} unknown"
    if row['n_with_age'] == 0:
        age = 'age n/a'
    else:
        age = (f"age {row['age_mean']:.1f} ± {row['age_sd']:.1f} y, median {row['age_median']:g} "
               f"[{row['age_min']:g}–{row['age_max']:g}]")
        if row['n_with_age'] < row['n']:
            age += f" (available for {row['n_with_age']}/{row['n']})"
    return f"{row['dataset'] + ':':<15} n = {row['n']:>3}, {sex}, {age}"


def print_sentence(rows):
    """
    Print the datasets description (n, females, age mean ± SD) to copy into an abstract; the trailing numbers are the
    reference numbers. Missing age is shown as XXXX.
    """
    def describe(dataset):
        row = next(r for r in rows if r['dataset'] == dataset)
        age = f"{row['age_mean']:.1f} ± {row['age_sd']:.1f}" if row['n_with_age'] else 'XXXX'
        return f"n={row['n']}, {row['n_female']} females, {age} y.o."

    print(f"\nThe following datasets were used: cervical open-access spine-generic multi-subject "
          f"({describe('spine-generic')}),7 whole-spine open-access dataset ({describe('whole-spine')}),8 lumbar "
          f"dataset from Balgrist University Hospital, Switzerland ({describe('Balgrist')}), an open-access lumbar "
          f"dataset EPFL, Switzerland ({describe('EPFL')}),9 and an open-access lumbar from Fudan University, China "
          f"({describe('Fudan')}).10")


def main():
    parser = argparse.ArgumentParser(description='Summarize sex and age of the subjects per dataset and pooled.')
    parser.add_argument('-o', default=OUTPUT_DIR, help=f'Output folder for the CSV files. Default: {OUTPUT_DIR}')
    args = parser.parse_args()
    dir_out = os.path.abspath(os.path.expanduser(args.o))
    os.makedirs(dir_out, exist_ok=True)

    df = pd.concat([
        load_dataset('spine-generic', load_participants_tsv(SPINE_GENERIC_TSV)),
        load_dataset('whole-spine', load_participants_tsv(WHOLE_SPINE_TSV)),
        load_dataset('Balgrist', load_balgrist(BALGRIST_XLSX)),
        load_dataset('EPFL', load_participants_tsv(EPFL_TSV)),
        load_dataset('Fudan', load_fudan(FUDAN_XLSX)),
    ], ignore_index=True)

    fname_subjects = os.path.join(dir_out, 'demographics_per_subject.csv')
    df.to_csv(fname_subjects, index=False)
    print(f'CSV file saved in {fname_subjects}.')

    rows = [summarize(df[df['dataset'] == dataset], dataset) for dataset in DATASET_YML]
    rows.append(summarize(df, 'Pooled'))
    df_summary = pd.DataFrame(rows)
    fname_summary = os.path.join(dir_out, 'demographics_across_datasets.csv')
    df_summary.to_csv(fname_summary, index=False)
    print(f'CSV file saved in {fname_summary}.')

    print('\n===== Sex and age (mean ± SD, median [range]) =====')
    for row in rows:
        print(format_row(row))
    print_sentence(rows)


if __name__ == '__main__':
    main()
