from scipy.stats import ttest_ind, levene
import os, glob, json
import numpy as np
import pandas as pd
import sys
from mne.stats import bonferroni_correction, fdr_correction
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
        _, t_pval_bf = bonferroni_correction(param_results.t_pval, alpha=0.05)
        param_results['t_pval_bf_corrected'] = t_pval_bf
        _, t_pval_bf = fdr_correction(param_results.t_pval, alpha=0.05)
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