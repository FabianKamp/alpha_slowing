import os, glob, json, sys
import numpy as np
import pandas as pd

def load_results(result_file, meta_file):
    # Load data
    results, meta_data = pd.read_csv(result_file), pd.read_csv(meta_file).rename(columns={'Unnamed: 0':'id', 'Age':'age', 'Gender_ 1=female_2=male':'gender'})
    # Create data dataframe
    results = meta_data[['id','age', 'gender']].merge(results)
    # Create age bin labels
    age_labels = {'20-25':1, '25-30':2, '30-35':3, '35-40':4, '55-60':5, '60-65':6, '65-70':7, '70-75':8, '75-80':9}
    results['age_bin'] = [age_labels[age] for age in results.age]
    results['age_group'] = [1 if age_bin<=4 else 2 for age_bin in results.age_bin]
    return results 