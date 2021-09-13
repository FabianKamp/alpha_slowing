import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm

# Helper Functions
def load_meta_data(meta_file):
    # Load meta data
    meta_data = pd.read_csv(meta_file)
    meta_data.rename(columns={'Unnamed: 0':'id'}, inplace=True)
    # Label Age bins
    age_labels = {'20-25':1, '25-30':2, '30-35':3, '35-40':4, '55-60':5, '60-65':6, '65-70':7, '70-75':8, '75-80':9}
    meta_data['age_bin'] = [age_labels[age] for age in meta_data.Age]
    return meta_data

def despine(ax):
    # Despine 
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False) # removes top and right spine
    # truncate the y spine
    ymin, ymax = ax.get_yticks()[1], ax.get_yticks()[-2]
    ax.spines['left'].set_bounds((ymin, ymax))
    xmin, xmax = ax.get_xticks()[0], ax.get_xticks()[-1]
    ax.spines['bottom'].set_bounds((xmin, xmax))

# Boxplot of alpha peaks
def plot_group_box(results, meta_file, param, fig, ax, cbar=False, show_subjects=False):
    meta_data = load_meta_data(meta_file)
    # Set up young and old group
    young = meta_data.loc[(meta_data.id.isin(results.id))&(meta_data.age_bin<=4), 'id']
    old = meta_data.loc[(meta_data.id.isin(results.id))&(meta_data.age_bin>=5), 'id']
    results = results.dropna(subset=[param]) 

    # Set up data + positions for the boxplot
    if show_subjects:
        positions = np.hstack([np.linspace(0.,0.4,len(young)), np.linspace(0.6,1,len(old))])
        data = [results.loc[results.id==sub, param] for sub in young] + [results.loc[results.id==sub, param] for sub in old]
        sensor_pos = [results.loc[results.id==sub, 'sensor_pos_a'] for sub in young] + [results.loc[results.id==sub, 'sensor_pos_a'] for sub in old]
        box_kwargs = dict(widths=0.05, medianprops=dict(linewidth=2., color='k'))
        ticklabel_kwargs = dict(labels = young.to_list()+old.to_list(), rotation=90)
        xlim = (-0.1,1.1)
        ax.text(s='Young Subjects', x=0.25, y=1, ha='center',transform=ax.transAxes)
        ax.text(s='Old Subjects', x=0.75, y=1, ha='center',transform=ax.transAxes)
    else:     
        positions=[0.2,0.8]
        data = [results.loc[results.id.isin(young),param], results.loc[results.id.isin(old),param]]
        sensor_pos = [results.loc[results.id.isin(young),'sensor_pos_a'],results.loc[results.id.isin(old),'sensor_pos_a']]
        box_kwargs = dict(widths=0.05, medianprops=dict(linewidth=2., color='k'))
        ticklabel_kwargs = dict(labels = ['Young', 'Old'])
        # Violin 
        vp = ax.violinplot(data, positions=positions, showextrema=False);
        for violin in vp['bodies']:
            violin.set_alpha(0.1)   
        xlim = (-0.1,1.1)
    
    # Boxplot
    ax.boxplot(data, positions=positions, showcaps=False, showfliers=False, notch=False, **box_kwargs);
    # Scatter    
    scatter_params = {'s':20, 'alpha':0.2}
    for pos, d, sensor in zip(positions,data,sensor_pos):
        ax.scatter(x=np.repeat(pos, len(d)), y=d, c=sensor, **scatter_params)
    # Colorbar
    if cbar:
        norm = matplotlib.colors.Normalize(vmin=-80, vmax=80)
        cmap = matplotlib.cm.get_cmap('viridis')
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), alpha=0.4, pad=0, fraction=0.025)
        ticks = cbar.get_ticks()[[0,-1]]
        cbar.set_ticks(ticks); 
        cbar.set_ticklabels(['Posterior','Anterior']) 
    # Axes + Ticks
    ax.set_xticklabels(**ticklabel_kwargs)
    ax.set_xlim(*xlim)
    despine(ax)
    return fig

def plot_model_fit(results, meta_file, ax, color_outliers=False):
    meta_data = load_meta_data(meta_file)
    # Set up young and old group
    young = meta_data.loc[(meta_data.id.isin(results.id))&(meta_data.age_bin<=4), 'id']
    old = meta_data.loc[(meta_data.id.isin(results.id))&(meta_data.age_bin>=5), 'id']
    # set up data
    data = [results.loc[results.id.isin(young),'model_error'], results.loc[results.id.isin(young),'model_rsquared'], 
            results.loc[results.id.isin(old),'model_error'], results.loc[results.id.isin(old),'model_rsquared']]
    # Scatter Settings
    positions = [0.1,0.3,0.7,0.9]
    ax.text(s='Young Subjects', x=0.2, y=ax.get_ylim()[-1]+0.1, ha='center', va='bottom')
    ax.text(s='Old Subjects', x=0.8, y=ax.get_ylim()[-1]+0.1, ha='center', va='bottom')
    scatter_params = {'s':20, 'alpha':0.2, 'c':'k'}
    # color settings
    if color_outliers:
        colors = cm.tab20.colors
        c = 0

    # Scatter plot
    for n, (pos, d) in enumerate(zip(positions,data)):
        d = d.to_numpy()
        x = np.random.randn(len(d))*0.01+pos
        ax.scatter(x=x, y=d, **scatter_params)
        
        # Color outliers
        if color_outliers:
            if n%2==0: outliers = np.sort(d)[-2:]
            else: outliers = np.sort(d)[:2]
            for n, outlier in enumerate(outliers): 
                outlier_idx = np.where(d == outlier)[0][0]
                sub_id = results.loc[(results.model_error==outlier)|(results.model_rsquared==outlier), 'id'].iloc[0]
                ax.scatter(x=x[outlier_idx], y=d[outlier_idx], s=40, label=sub_id, color=colors[c])
                c+=1
    ax.set_xticks(positions)
    ax.set_xticklabels(['Error', 'R squared']*2)
    ax.set_yticks([0.,0.1,0.2,0.5,0.8,0.9,1])
    ax.grid()    
    if color_outliers:
        ax.legend(bbox_to_anchor=(1,1))

def plot_params(param_dict, ax):
    s = '\n'.join([f'{k}:{val}' for k,val in param_dict.items()])
    ax.text(s=s,x=0.5,y=0.5,transform=ax.transAxes, ha='center', va='center', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_visible(False) # removes top and right spine


