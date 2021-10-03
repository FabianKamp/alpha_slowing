import os, glob, json, sys
import numpy as np
import pandas as pd

def load_results(result_file, meta_file, age_file=None, name_match_file=None):
    # Load data
    results, meta_data = pd.read_csv(result_file), pd.read_csv(meta_file).rename(columns={'Unnamed: 0':'id', 'Age':'age', 'Gender_ 1=female_2=male':'gender'})
    # Create data dataframe
    results = meta_data[['id','age', 'gender']].merge(results)
    # Create age bin labels
    age_labels = {'20-25':1, '25-30':2, '30-35':3, '35-40':4, '55-60':5, '60-65':6, '65-70':7, '70-75':8, '75-80':9}
    results['age_bin'] = [age_labels[age] for age in results.age]
    results['age_group'] = [1 if age_bin<=4 else 2 for age_bin in results.age_bin]

    # Add exact ages to dataframe
    if age_file and name_match_file:
        name_match = pd.read_csv(name_match_file)
        ages = pd.read_csv(age_file)
        ages = ages.merge(name_match, left_on='ID', right_on='Initial_ID')[['Initial_ID', 'INDI_ID', 'Age']]\
                   .rename(columns={'Initial_ID':'initial_id', 'INDI_ID':'id', 'Age':'age_years'})
        results = results.merge(ages, on='id')

    return results 

def load_meta_data(meta_file):
    """
    Loads meta data and adds columns with age_bin and age_group
    """
    # Load meta data
    meta_data = pd.read_csv(meta_file)
    meta_data.rename(columns={'Unnamed: 0':'id'}, inplace=True)
    # Label Age bins
    age_labels = {'20-25':1, '25-30':2, '30-35':3, '35-40':4, '55-60':5, '60-65':6, '65-70':7, '70-75':8, '75-80':9}
    meta_data['age_bin'] = [age_labels[age] for age in meta_data.Age]
    meta_data['age_group'] = [1 if age_bin<=4 else 2 for age_bin in meta_data.age_bin]
    return meta_data

def get_ids(group_folder, meta_file, group_size):
    """
    Returns id list of old and young group for which there is data in the group folder.
    The number of ids per group is defined as group_size.
    """
    # Get downloaded data ids
    data_files = glob.glob(group_folder + '**\\*EC.set')
    subject_ids = [file.split('\\')[-2] for file in data_files]

    # Load meta data
    meta_data = load_meta_data(meta_file)

    # Young subjects
    young = meta_data.loc[(meta_data['id'].isin(subject_ids))&(meta_data.age_group==1), 'id']
    young = young.iloc[:group_size].to_list()

    # Old subjects
    old = meta_data.loc[(meta_data['id'].isin(subject_ids))&(meta_data.age_group>=2), 'id']
    old = old.iloc[:group_size].to_list()

    return {'young': young, 'old':old}

def get_fit_outliers(results, thresh_rsq=0.9, thresh_err=0.1):
    """
    Returns array of subjects that have model fits over/under the thresholds.
    """ 
    if type(results)!=pd.DataFrame: 
        results = pd.read_csv(results)
    outliers = results.loc[(results.model_error>=thresh_err)|(results.model_rsquared<=thresh_rsq), 'id']
    return outliers.unique()

def count_missing_param_values(results, param=None):
    # Count missing values per subject
    ch_names = results.ch_names.unique()
    ids = results['id'].unique()
    if param:
        w_results = results.pivot(index=['id'], columns='ch_names', values=param)
    else:
        w_results = results.pivot(index=['id'], columns='ch_names', values='ch_names')
    ids_count = [[i, np.sum(w_results.loc[i].isna())] for i in np.sort(ids) if np.any(w_results.loc[i].isna())]
    chs_count = [[c, np.sum(w_results[c].isna())] for c in np.sort(ch_names) if np.any(w_results[c].isna())]
    return pd.DataFrame(ids_count, columns=['id', 'count']), pd.DataFrame(chs_count, columns=['ch_names', 'count'])