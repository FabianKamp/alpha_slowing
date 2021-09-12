import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages


def load_meta_data(meta_file):
    # Load meta data
    meta_data = pd.read_csv(meta_file)
    meta_data.rename(columns={'Unnamed: 0':'id'}, inplace=True)
    # Label Age bins
    age_labels = {'20-25':1, '25-30':2, '30-35':3, '35-40':4, '55-60':5, '60-65':6, '65-70':7, '70-75':8, '75-80':9}
    meta_data['age_bin'] = [age_labels[age] for age in meta_data.Age]
    return meta_data

def plot_group_peak_cf(results, meta_file, fig, show_subjects=False):
    meta_data = load_meta_data(meta_file)
    # Set up young and old group
    young = meta_data.loc[(meta_data.id.isin(results.id))&(meta_data.age_bin<=4), 'id']
    old = meta_data.loc[(meta_data.id.isin(results.id))&(meta_data.age_bin>=5), 'id']

    results = results.dropna(subset=['alpha_peak_cf'])
    young_peaks = results.loc[results.id.isin(young),'alpha_peak_cf']
    young_pos = results.loc[results.id.isin(young),'sensor_pos_a']
    old_peaks = results.loc[results.id.isin(old),'alpha_peak_cf']
    old_pos = results.loc[results.id.isin(old),'sensor_pos_a']

    # Get axes
    ax = fig.get_axes()[0]
    if show_subjects:
        data = [results.loc[results.id==sub, 'alpha_peak_cf'] for sub in young] + [results.loc[results.id==sub, 'alpha_peak_cf'] for sub in old]
        sensor_pos = [results.loc[results.id==sub, 'sensor_pos_a'] for sub in young] + [results.loc[results.id==sub, 'sensor_pos_a'] for sub in old]
        positions = np.hstack([np.linspace(0.,0.4,len(young)), np.linspace(0.6,1,len(old))])
        medianprops = dict(linewidth=2., color='k')
        ax.boxplot(data, positions=positions, showcaps=False, showfliers=False, widths=0.02, medianprops=medianprops, notch=True);
        ax.text(s='Young Subjects', x=0.2, y=ax.get_ylim()[-1], ha='center')
        ax.text(s='Old Subjects', x=0.8, y=ax.get_ylim()[-1], ha='center')

        # Scatter
        scatter_params = {'s':20, 'alpha':0.2}
        for pos, d, sensor in zip(positions,data,sensor_pos):
            ax.scatter(x=np.repeat(pos, len(d)), y=d, c=sensor, **scatter_params)

        ax.set_xticklabels(young.to_list()+old.to_list(), rotation=90)
        ax.set_xlim(-0.1,1.1)

        norm = matplotlib.colors.Normalize(vmin=-80, vmax=80)
        cmap = matplotlib.cm.get_cmap('viridis')
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), alpha=0.4, pad=0, fraction=0.025)
        ticks = cbar.get_ticks()[[0,-1]]
        cbar.set_ticks(ticks); 
        cbar.set_ticklabels(['Posterior','Anterior'])

    else:     
        # Boxplot
        data = [young_peaks, old_peaks]
        labels = ['Young', 'Old']
        positions=[0.2,0.8]
        medianprops = dict(linewidth=2., color='k')
        ax.boxplot(data, positions=positions, showcaps=False, showfliers=False, widths=0.05, medianprops=medianprops, notch=True);
        ax.set_xticklabels(labels);

        # Violin 
        vp = ax.violinplot(data, positions=positions, showextrema=False);
        for violin in vp['bodies']:
            violin.set_alpha(0.1)

        # Scatter
        scatter_params = {'s':20, 'alpha':0.2}
        cf = ax.scatter(x=np.repeat(positions[0], len(young_peaks)), y=young_peaks, c=young_pos, **scatter_params)
        ax.scatter(x=np.repeat(positions[1], len(old_peaks)), y=old_peaks, c=old_pos, **scatter_params)
        cbar = fig.colorbar(cf, pad=0, fraction=0.025)
        cbar.set_ticks([-80,0,80]); 
        cbar.set_ticklabels(['Posterior','Central','Anterior'])

    # Despine 
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False) # removes top and right spine
    # truncate the y spine
    ymin, ymax = ax.get_yticks()[1], ax.get_yticks()[-2]
    ax.spines['left'].set_bounds((ymin, ymax))
    xmin, xmax = ax.get_xticks()[0], ax.get_xticks()[-1]
    ax.spines['bottom'].set_bounds((xmin, xmax))
    return fig