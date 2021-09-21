from scipy.stats import ttest_ind, levene
import os, glob, json, sys
import numpy as np
import pandas as pd
import mne
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
sys.path.append('C:/Users/Kamp/Documents/nid/scripts')

def get_group_stats(data, params): 
    """
    Takes mean over all channels for each subject and calculates over all parameters
        * mean difference
        * t-statistics
        * levene statistics
        * welch t'test
    between the old and young group. 
    The data must contain the column "age_group". 

    :params data - dataframe resulting from the group pipeline containing old and young subjects
    :params list - params in the data for which the stats are calculated
    :returns pd.DataFrame with the statistics
    """
    assert type(params)==list, "Params must be a list."
    assert 'age_group' in data.columns, "age_group not in dataframe."
    results = []
    # Loop over parameters
    for param in params:
        assert param in data.columns, f"{param} not in dataframe."
        data_nonan = data.dropna(subset=[param])
        # Take subject mean over all channels for each subject
        sub_means = data_nonan[['id','age_group', param]].groupby('id').mean().reset_index()
        # Get young and old sample
        young_sample = sub_means.loc[sub_means.age_group==1, param]
        old_sample = sub_means.loc[sub_means.age_group==2, param]
        # Get stats
        mean_diff = np.mean(young_sample) - np.mean(old_sample)
        t_stat, t_pval = ttest_ind(a=young_sample,b=old_sample)
        welch_stat, welch_pval = ttest_ind(a=young_sample,b=old_sample, equal_var=False)
        levene_stat, levene_pval = levene(young_sample, old_sample)
        # Param stats
        results.append({'param':param, 'mean_diff': mean_diff,
                        't_stat':t_stat, 't_pval':t_pval,
                        'welch_stat':welch_stat, 'welch_pval':welch_pval,
                        'levene_stat':levene_stat, 'levene_pval':levene_pval})
    return pd.DataFrame(results)

def get_sensor_stats(data, params): 
    """
    Iterates over all parameters and calculates
    for each sensor 
        * mean difference
        * t-statistics
        * levene statistics
        * welch t'test
    between the old and young group of subjects with regard to the parameter. 
    The data must contain the column "age_group". 

    :params data - dataframe resulting from the group pipeline containing old and young subjects
    :params list - params in the data for which the stats are calculated
    :returns pd.DataFrame with the statistics
    """
    assert type(params)==list, "Params must be a list."
    assert 'age_group' in data.columns, "age_group not in dataframe."
    results = []
    # Loop over parameters
    for param in params:
        assert param in data.columns, f"{param} not in dataframe."
        data_nonan = data.dropna(subset=[param])
        param_results = []
        # Loop over all sensors
        for channel in data_nonan.ch_names.unique(): 
            # Get samples
            sensor_data = data_nonan.loc[data_nonan.ch_names==channel]
            young_sample = sensor_data.loc[sensor_data.age_group==1, param]
            old_sample = sensor_data.loc[sensor_data.age_group==2, param]
            # Get stats
            mean_diff = np.mean(young_sample) - np.mean(old_sample)
            t_stat, t_pval = ttest_ind(a=young_sample,b=old_sample)
            welch_stat, welch_pval = ttest_ind(a=young_sample,b=old_sample, equal_var=False)
            levene_stat, levene_pval = levene(young_sample, old_sample)
            # Param stats
            param_results.append({'param':param, 'ch_names':channel, 'mean_diff': mean_diff,
                                  't_stat':t_stat, 't_pval':t_pval,
                                  'welch_stat':welch_stat, 'welch_pval':welch_pval,
                                  'levene_stat':levene_stat, 'levene_pval':levene_pval})
        param_results = pd.DataFrame(param_results)
        # Multiple comparison correction
        _, t_pval_bf = mne.stats.bonferroni_correction(param_results.t_pval, alpha=0.05)
        param_results['t_pval_bf_corrected'] = t_pval_bf
        _, t_pval_bf = mne.stats.fdr_correction(param_results.t_pval, alpha=0.05)
        param_results['t_pval_fdr_corrected'] = t_pval_bf

        # Save to results
        results.append(param_results)
    return pd.concat(results)

def get_descr_sensor_stats(data, params):
    """
    Iterate over all sensors and calculate group mean + variance 
    for each age group with respect to the parameter.
    :params data - dataframe resulting from the group pipeline containing old and young subjects
    :params param - parameter in the data for which the stats are calculated
    :returns pd.DataFrame describtive statistics
    """
    assert type(params)==list, "Params must be a list."
    assert 'age_group' in data.columns, "age_group not in dataframe."
    results = []
    # Loop over parameters
    for param in params:
        assert param in data.columns, f"{param} not in dataframe."
        data_nonan = data.dropna(subset=[param])
        # Loop over all sensors
        for channel in data_nonan.ch_names.unique(): 
            sensor_data = data_nonan.loc[data_nonan.ch_names==channel]
            # Old sample 
            old_sample = sensor_data.loc[sensor_data.age_group==2, param]
            results.append({'age_group':2, 'param':param, 'ch_names':channel, 
                            'mean': np.mean(old_sample),'sd':np.std(old_sample),
                            'max':np.max(old_sample),'min':np.min(old_sample)})
            # Young sample
            young_sample = sensor_data.loc[sensor_data.age_group==1, param]
            results.append({'age_group':1, 'param':param, 'ch_names':channel, 
                            'mean': np.mean(young_sample),'sd':np.std(young_sample),
                            'max':np.max(young_sample),'min':np.min(young_sample)})
    results = np.round(pd.DataFrame(results),3)
    return results

def get_adjacency(ch_names): 
    """
    Returns the adjacency mat for input channel names using the standard_1005 montage
    """
    montage = mne.channels.make_standard_montage(kind='standard_1005')
    info = mne.create_info(ch_names, sfreq=2500, ch_types='eeg')
    info.set_montage(montage)
    adj, ch_names = mne.channels.find_ch_adjacency(info, ch_type='eeg')
    return adj

def permutation_cluster_test(results, param, pval): 
    """
    Performs permutation cluster test (t-test) over sensors for input parameter.
    Missing data is replaced with the sensor mean across all subjects.
    Returns clusters of sensors and corresponding p values

    :param results - dataframe containing the results from the group-pipeline
    :param param - string, column used for permutation cluster test over sensors
    :param pval - pval that set the threshold for permutation cluster test
    :returns ch_names, t-statistics, masks, p_vals
    """
    ch_names = list(results.ch_names.unique())
    wide_data = results.pivot(index=['age_group','id'], columns='ch_names', values=param)[ch_names]
    age_groups = wide_data.index.get_level_values(0)
    sub_ids = wide_data.index.get_level_values(1)
    # Replace missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imputer.fit_transform(wide_data)
    # Split into young and old group
    young_X = X[age_groups==1]
    old_X = X[age_groups==2]
    # Get adjacency matrix
    adj = get_adjacency(ch_names)
    # Compute permutation test
    thres = scipy.stats.t.ppf(1-pval/2, df=len(sub_ids)-1)
    results = mne.stats.permutation_cluster_test([young_X, old_X], adjacency=adj, threshold=thres, 
                                                  stat_fun=mne.stats.ttest_ind_no_p, out_type='mask', tail=0)
    # Results
    ch_names = np.array(ch_names)
    statistic = results[0]
    masks = results[1]
    p_vals = results[2]
    return ch_names, statistic, masks, p_vals

def logistic_regression(results, param): 
    """
    Performs a logistic regression. X is the matrix (n_subjects, parameter value for each channel) and y is the age_group array (n_subjects,). 
    Missing data is replaced by the channel mean.
    """
    # Transfrom dataframe to wide format
    wide_data = results.pivot(index=['id', 'age_group'], columns='ch_names', values=param)
    # Replace missing data and set up X,y
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imputer.fit_transform(wide_data)
    y = wide_data.index.get_level_values('age_group').to_numpy()
    # Logistic reg
    log_reg = LogisticRegression(solver='liblinear', penalty='l2', max_iter=1000)
    log_reg.fit(X,y)
    return log_reg, X, y