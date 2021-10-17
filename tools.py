import os, glob, json, sys
import numpy as np
import pandas as pd
import mne, scipy

def load_meta_data(meta_file, age_file=None, name_match_file=None, columns=None):
    """
    Loads meta data and adds columns with age_bin, age_group and exact age if age_file is provided
    """
    # Load meta data
    meta_data = pd.read_csv(meta_file)
    meta_data.rename(columns={'Unnamed: 0':'id', 'Age':'age', 'Gender_ 1=female_2=male':'gender'}, inplace=True)
    # Label Age bins
    age_labels = {'20-25':1, '25-30':2, '30-35':3, '35-40':4, '55-60':5, '60-65':6, '65-70':7, '70-75':8, '75-80':9}
    meta_data['age_bin'] = [age_labels[age] for age in meta_data.age]
    meta_data['age_group'] = [1 if age_bin<=4 else 2 for age_bin in meta_data.age_bin]
    if columns is None:
        columns = ['id', 'age', 'age_bin', 'age_group', 'gender']
    assert type(columns)==list, 'Columns must be a list of column names'
    meta_data = meta_data[columns] # Load only some columns
    # Add exact ages to dataframe
    if age_file and name_match_file:
        name_match = pd.read_csv(name_match_file)
        ages = pd.read_csv(age_file)
        ages = ages.merge(name_match, left_on='ID', right_on='Initial_ID')[['Initial_ID', 'INDI_ID', 'Age']]\
                   .rename(columns={'Initial_ID':'initial_id', 'INDI_ID':'id', 'Age':'age_years'})
        meta_data = meta_data.merge(ages, on='id')
    return meta_data

def load_results(result_file, meta_file, age_file=None, name_match_file=None):
    # Load data
    results = pd.read_csv(result_file) 
    meta_data = load_meta_data(meta_file, age_file, name_match_file)
    # Create choose columns from meta data to merge into results
    meta_columns = ['id', 'age', 'age_bin', 'age_group', 'gender']
    if age_file and name_match_file:
        meta_columns.extend(['initial_id', 'age_years'])
    results = meta_data[meta_columns].merge(results)
    return results 

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

def count_channels(data_folder=None, out_file=None, verbose=False):
    """
    Returns Dataframe showing the valid channels for each subject.
    """
    if data_folder is None:
        data_folder = 'D:\\nid\\lemon-dataset\\eeg-preprocessed\\'
    subjects = os.listdir(data_folder)
    channel_dict = {}
    all_channels = []
    for n, subject in enumerate(subjects):
        if verbose: 
            print(f"Loading  {subject}")
        subject_folder = os.path.join(data_folder, subject)
        data_file = os.path.join(subject_folder, f'{subject}_EC.set')
        channels = get_channel_names(data_file)
        all_channels.extend(list(channels))
        channel_dict.update({subject:channels})        
    all_channels = np.unique(all_channels)
    # Check for each subject which channels of all_channels are present
    count = [[sub] + [int(ch in sub_channels) for ch in all_channels] for sub, sub_channels in channel_dict.items()]
    count_df = pd.DataFrame(count, columns=['id']+list(all_channels)).set_index('id')
    # Add counts
    count_df.loc['subject_total',:] = count_df.sum(axis=0)
    count_df.loc[:,'channel_total'] = count_df.sum(axis=1)
    if out_file: 
        count_df.to_csv(out_file, index=True)
    return count_df

def get_channel_names(data_file):
    raw = mne.io.read_raw_eeglab(data_file, verbose=False)
    return raw.ch_names

def convert2mat(data_file, out_file, channels=None, crop=None):
    file_ending = data_file.split('\\')[-1]
    subject_id, condition = (file_ending.split('.')[0]).split('_')
    raw = mne.io.read_raw_eeglab(data_file, verbose=False)
    # Exclude channels and crop dataset
    if crop:
        raw.crop(tmin=crop[0], tmax=crop[1])
    if channels:
        raw.pick(channels)
    data,time = raw[:]
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    mdict = dict(data=data, time=time, subject_id= subject_id, condition=condition, 
                 sfreq=sfreq, ch_names=ch_names)
    scipy.io.savemat(out_file, mdict)

def sample_convert2mat(sel_name):
    # Load meta-data
    meta_file = 'D:\\nid\\lemon-dataset\\Behavioural_Data_MPILMBB_LEMON\\META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
    age_file = 'D:\\nid\\lemon-dataset\\Behavioural_Data_MPILMBB_LEMON\\LEMON_BIDS_ID_AGE.csv'
    name_match_file = 'D:\\nid\\lemon-dataset\\name_match.csv'
    meta_data = load_meta_data(meta_file, age_file=age_file, name_match_file=name_match_file)

    # Get sample subjects and channels
    sample_file = 'D:\\nid\\lemon-dataset\\samples.json'
    with open(sample_file, 'r') as file: 
        sample_dict = json.load(file)
    ids = sample_dict[sel_name]['subjects']
    channels = sample_dict[sel_name]['ch_names']

    # Check group sizes
    old = len(meta_data.loc[(meta_data['id'].isin(ids))&(meta_data['age_group']==2)])
    young = len(meta_data.loc[(meta_data['id'].isin(ids))&(meta_data['age_group']==1)])
    print('Old Group: ', old, '\nYoung Group: ', young)

    # Data folder
    data_folder = 'D:\\nid\\lemon-dataset\\eeg-preprocessed'
    # Out folder
    out_folder = f'D:\\nid\\mat_files_{sel_name}'
    if not os.path.isdir(out_folder): 
        os.mkdir(out_folder)
    
    for subject in ids: 
        subject_folder = os.path.join(data_folder, subject)
        data_file = os.path.join(subject_folder, f'{subject}_EC.set')
        out_file = os.path.join(out_folder, f'{subject}.mat')
        convert2mat(data_file, out_file, channels=channels)

def compute_margins(df): 
    if df.index[-1]=='subject_total':
        df.loc['subject_total',:] = df.iloc[:-1,:].sum(axis=0)
    else: 
        df.loc['subject_total',:] = df.iloc[:,:].sum(axis=0)
    if df.columns[-1]=='channel_total':
        df.loc[:,'channel_total'] = df.iloc[:,:-1].sum(axis=1)
    else: 
        df.loc[:,'channel_total'] = df.iloc[:,:].sum(axis=1)

def exclude_channels(count, max_missing):
    # Exclude all Temporal channels
    excluded_channels = [ch for ch in count.columns[:-1] if ch.startswith('T')]
    # Exclude channels that lack more than max_missing subjects
    exclude_idx = np.where(count.loc['subject_total']<len(count)-max_missing-1)[0]
    excluded_channels.extend(count.columns[exclude_idx])
    # Exclude
    count = count[[ch for ch in count.columns if ch not in excluded_channels]].copy()
    compute_margins(count)
    # Exclude Subjects that don't have all channels
    count = count.loc[count.channel_total==np.max(count.channel_total.iloc[:-1])].copy()
    compute_margins(count)
    return count, set(excluded_channels)

def create_sample(sel_name, count_file, selection_file, out_file=None):
    """
    Takes the name of the channel selection in the selection file and creates a sample of subjects that posses all channels. 
    If out_file is given saves the sample to json file.
    returns list of subject, list of channels
    """
    count = pd.read_csv(count_file, index_col=0)
    count.rename(columns={'total':'channel_total'}, index={'total':'subject_total'}, inplace=True)
    with open(selection_file, 'r') as file: 
        sel_dict = json.load(file)
    # Select Channels and recompute margins
    count = count[sel_dict[sel_name]]
    compute_margins(count)
    # Select Subjects and recompute margins
    count = count.loc[count.channel_total==len(sel_dict[sel_name])]
    compute_margins(count)
    
    # Save sample file
    ch_names, subjects = sorted(count.columns[:-1].to_list()), count.index[:-1].to_list()
    if out_file:
        sample_dict = dict(ch_names=ch_names, subjects=subjects)
        with open(out_file, 'w') as file: 
            json.dump(sample_dict, file)            
    return subjects, ch_names

def create_sample_meta(sel_name):
    folder = 'D:\\nid\\lemon-dataset'
    meta_file = os.path.join(folder, 'Behavioural_Data_MPILMBB_LEMON', 'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')
    name_match = os.path.join(folder, 'name_match.csv')
    age_file = os.path.join(folder, 'Behavioural_Data_MPILMBB_LEMON', 'LEMON_BIDS_ID_AGE.csv')
    meta_data = load_meta_data(meta_file, age_file=age_file, name_match_file=name_match)

    sample_file = os.path.join(folder, 'samples.json')
    with open(sample_file, 'r') as file: 
        sample_dict = json.load(file)
    subjects = sample_dict[sel_name]['subjects']
    print(len(subjects))
    meta_data = meta_data.loc[meta_data['id'].isin(subjects)]
    meta_file = os.path.join(folder, 'meta_files', f'meta_{sel_name}.csv')
    meta_data.to_csv(meta_file, index=False) 
