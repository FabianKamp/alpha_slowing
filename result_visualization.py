import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib import cm
import mne

# Helper Functions
def load_meta_data(meta_file):
    # Load meta data
    meta_data = pd.read_csv(meta_file)
    meta_data.rename(columns={'Unnamed: 0':'id'}, inplace=True)
    # Label Age bins
    age_labels = {'20-25':1, '25-30':2, '30-35':3, '35-40':4, '55-60':5, '60-65':6, '65-70':7, '70-75':8, '75-80':9}
    meta_data['age_bin'] = [age_labels[age] for age in meta_data.Age]
    meta_data['age_group'] = [1 if age_bin<=4 else 2 for age_bin in meta_data.age_bin]
    return meta_data

def get_group_ids(results, meta_file):
    """
    Return young and old subject lists from meta data file
    """
    meta_data = load_meta_data(meta_file)
    # Set up young and old group
    young = meta_data.loc[(meta_data.id.isin(results.id))&(meta_data.age_bin<=4), 'id'].to_list()
    old = meta_data.loc[(meta_data.id.isin(results.id))&(meta_data.age_bin>=5), 'id'].to_list()
    return young, old

def despine(axes):
    # if input is only one axes
    if isinstance(axes,matplotlib.axes.Axes):
        axes = [axes]
    # iter overall axes and despine
    for ax in axes:
        # Despine 
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False) # removes top and right spine
        # truncate the y spine
        ymin, ymax = ax.get_yticks()[1], ax.get_yticks()[-2]
        ax.spines['left'].set_bounds((ymin, ymax))
        xmin, xmax = ax.get_xticks()[0], ax.get_xticks()[-1]
        ax.spines['bottom'].set_bounds((xmin, xmax))

def get_standard_eeg_positions(ch_names):
    """
    Loads the 2D standard_1005 positions from file. 
    Indexes only those channels in ch_names and returns the xy positions for plots
    :params channel names
    :returns np.array - x and y postion for each channel
    """
    standard_1005_file = 'C:/Users/Kamp/Documents/nid/standard_1005_2D.txt' # from eeg_positions repository
    standard_1005_positions = pd.read_csv(standard_1005_file, sep='\t', index_col=0)
    positions = []
    for c in ch_names:
        positions.append(standard_1005_positions.loc[c].to_numpy())
    return np.vstack(positions)

# Boxplot of alpha peaks
def plot_group_box(results, param, fig, ax, cbar=False, show_subjects=False, boxwidths=0.05):
    # Set up young and old group
    young = results.loc[results.age_group==1, 'id'].unique()
    old = results.loc[results.age_group==2, 'id'].unique()
    results = results.dropna(subset=[param]) 
    # Set up data + positions for the boxplot
    if show_subjects:
        positions = np.hstack([np.linspace(0.,0.4,len(young)), np.linspace(0.6,1,len(old))])
        data = [results.loc[results.id==sub, param] for sub in young] + [results.loc[results.id==sub, param] for sub in old]
        sensor_pos = [results.loc[results.id==sub, 'sensor_pos_a'] for sub in young] + [results.loc[results.id==sub, 'sensor_pos_a'] for sub in old]
        box_kwargs = dict(widths=boxwidths, medianprops=dict(linewidth=2., color='k'))
        ticklabel_kwargs = dict(labels = list(young)+list(old), rotation=90)
        xlim = (-0.1,1.1)
        ax.text(s='Young Subjects', x=0.25, y=1, ha='center',transform=ax.transAxes)
        ax.text(s='Old Subjects', x=0.75, y=1, ha='center',transform=ax.transAxes)
    else:     
        positions=[0.2,0.8]
        data = [results.loc[results.age_group==1,param], results.loc[results.age_group==2,param]]
        sensor_pos = [results.loc[results.age_group==1,'sensor_pos_a'],results.loc[results.age_group==2,'sensor_pos_a']]
        box_kwargs = dict(widths=boxwidths, medianprops=dict(linewidth=2., color='k'))
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
            if n%2==0: outliers = np.sort(d[~np.isnan(d)])[-2:]
            else: outliers = np.sort(d[~np.isnan(d)])[:2]
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

def plot_box(results, params, labels, ax,  add_violin=True, add_strips=True): 
    # Remove nan values 
    results = results.dropna(subset=params)
    data = [results[param] for param in params]
    # Boxplot
    positions=np.arange(len(params))
    medianprops = dict(linewidth=2., color='tab:red')
    ax.boxplot(data, positions=positions, showcaps=False, showfliers=False, widths=0.25, medianprops=medianprops)
    # Add violin
    if add_violin:
        vp = ax.violinplot(data,positions,showextrema=False)
        for violin in vp['bodies']:
            violin.set_alpha(0.2) 
    # Add lines 
    if add_strips:
        for d, position in zip(data,positions):
            ax.scatter(x=np.repeat(position, len(d)), y=d, alpha=0.02, marker="_", color='k') 
    ax.set_xticklabels(labels)
    #despine(ax)

def plot_sub_means(results, param, ax, title='', annotate=False, **boxkwargs):
    """
    Plots the mean values of the param column for each subject. 
    Splits age_group in seperate boxplots.
    :param results - pd.DataFrame
    :param param - column in pd.Dataframe which is plotted
    :param ax - axes
    :param annotate - bool, if true plots the subject ids
    :returns ax
    """
    assert 'age_group' in results.columns, 'age_group must be columns of dataframe.'
    assert param in results.columns, f'{param} must be columns of dataframe.'
    assert 'id' in results.columns, 'id must be columns of dataframe.'
    
    # Get subject means
    results = results.dropna(subset=[param])
    means = results.groupby('id').mean()[[param, 'age_group']].reset_index()
    young_sample, old_sample = means[param].loc[means.age_group==1], means[param].loc[means.age_group==2]
    positions = [1,2]
    # Box + Violin
    ax.boxplot([young_sample, old_sample], positions=positions, showcaps=False, showfliers=False, medianprops=dict(linewidth=2., color='tab:red'), **boxkwargs)
    vp = ax.violinplot([young_sample, old_sample], positions=positions, showextrema=False)
    for violin in vp['bodies']:
        violin.set_alpha(0.2) 
    if annotate:
        for sub in means.id: 
            y = means.loc[means.id==sub, param]
            x = means.loc[means.id==sub, 'age_group']
            ax.annotate(text=sub, xy=(x,y), size=5, ha='center', va='center')
    # Scatter
    else: 
        scatterkwargs = dict(marker='o', color='white', edgecolor='k', linewidth=0.5, alpha=0.8, s=60)
        ax.scatter(x=np.ones(len(young_sample))*positions[0],y=young_sample, 
                   **scatterkwargs)
        ax.scatter(x=np.ones(len(old_sample))*positions[1],y=old_sample, 
                   **scatterkwargs)
    ax.set_xticklabels(['Young', 'Old'])
    ax.set_title(title)
    return ax

# Topomaps
def plot_topomap(ch_names, data, ax, cax, **mne_kwargs):
    positions = get_standard_eeg_positions(ch_names)
    if ('vmin' not in mne_kwargs.keys()) | ('vmax' not in mne_kwargs.keys()):
        mne_kwargs.update({'vmin':np.min(data),'vmax':np.max(data)})
    im,cn = mne.viz.plot_topomap(data, positions, axes=ax, names=ch_names,  sphere=1,
                                 show=False, **mne_kwargs);
    cbar = plt.colorbar(im, cax, orientation='horizontal')
    # Add inner circle
    head_shape = plt.Circle((0, 0), 0.72658518, color="k", fill=False, linewidth=0.5)
    ax.add_artist(head_shape)
    # Add lines
    ax.vlines(x=0, ymin=-1, ymax=1, color="black", linewidth=0.5)
    ax.hlines(y=0, xmin=-1, xmax=1, color="black", linewidth=0.5)
    return ax, cbar

def plot_topo_means(data, title, group, **mne_kwargs):
    """
    Takes as input dataframe of descriptive statistics. 
    Indexes only the group of interest (1|2). 
    Plots topomaps of the MEAN 
        * alpha peak frequency
        * alpha peak cf
        * spectral centroid
    :param data - pd.DataFrame with descr stats
    :param title - figure title
    :param group - age_group 1|2
    :param **mne_kwargs - keyword arguments that are passed to mne.viz.plot_topomap    
    :returns figure
    """
    data = data.loc[data.age_group==group].copy()
    fig = plt.figure(figsize=(18,8), tight_layout=True)
    fig.suptitle(title, fontsize=25)
    gs = gridspec.GridSpec(20, 45)
    vmin, vmax = np.min(data['mean']),np.max(data['mean'])
    # Plot alpha peak frequency
    ax_topo = fig.add_subplot(gs[:-1,:15])
    ax_cbar = fig.add_subplot(gs[-1,3:12])
    p_data = data.loc[data.param=='alpha_peak_freqs', 'mean']
    ch_names = data.loc[data.param=='alpha_peak_freqs', 'ch_names']
    ax_topo, cbar = plot_topomap(ch_names=ch_names, data=p_data, ax=ax_topo, cax=ax_cbar, 
                                 vmin=vmin, vmax=vmax, **mne_kwargs)
    ax_topo.set_title('Mean Alpha Peak Frequency', fontsize=20)
    ax_cbar.tick_params(labelsize=20)
    # Plot alpha cf
    ax_topo = fig.add_subplot(gs[:-1,15:30])
    ax_cbar = fig.add_subplot(gs[-1,18:27])
    p_data = data.loc[data.param=='alpha_peak_cf', 'mean']
    ch_names = data.loc[data.param=='alpha_peak_cf', 'ch_names']
    ax_topo, cbar = plot_topomap(ch_names=ch_names, data=p_data, ax=ax_topo, cax=ax_cbar, 
                                 vmin=vmin, vmax=vmax, **mne_kwargs)
    ax_topo.set_title('Mean Alpha Peak CF', fontsize=20)
    ax_cbar.tick_params(labelsize=20)
    # Plot alpha sc
    ax_topo = fig.add_subplot(gs[:-1,30:])
    ax_cbar = fig.add_subplot(gs[-1,33:42])
    p_data = data.loc[data.param=='spectral_centroid', 'mean']
    ch_names = data.loc[data.param=='spectral_centroid', 'ch_names']
    ax_topo, cbar = plot_topomap(ch_names=ch_names, data=p_data, ax=ax_topo, cax=ax_cbar, 
                                 vmin=vmin, vmax=vmax, **mne_kwargs)
    ax_topo.set_title('Mean Spectral Centroid', fontsize=20)
    ax_cbar.tick_params(labelsize=20)
    plt.subplots_adjust(hspace=0)
    return fig

def plot_topo_stats(data, param, title, **mne_kwargs):
    """
    Takes as input dataframe of group difference statistics. 
    Plots topomaps of the 
        * mean diff
        * t'test pvalue
    :param data - pd.DataFrame with descr stats
    :param param - str parameter, 'alpha_peak_frequency'|'alpha_peak_cf'|'spectral_centroid'
    :param title - figure title
    :param **mne_kwargs - keyword arguments that are passed to mne.viz.plot_topomap    
    :returns figure
    """
    assert param in data.param.to_list(), f'Param {param} not in dataframe param column.'
    data = data.loc[data.param==param].copy()
    fig = plt.figure(figsize=(9,6), tight_layout=True)
    fig.suptitle(title,fontsize=25)
    gs = gridspec.GridSpec(15, 20)
    # Mean Difference
    ch_names = data.ch_names.unique()
    ax_topo = fig.add_subplot(gs[:-1,:10])
    ax_cbar = fig.add_subplot(gs[-1,2:8])
    ax_topo, cbar = plot_topomap(ch_names=ch_names, data=data['mean_diff'], ax=ax_topo, cax=ax_cbar, 
                                 show_names=True, cmap='RdBu_r', **mne_kwargs)
    cbar.ax.tick_params(labelsize=15)
    ax_topo.set_title('Mean Difference', fontsize=20)
    # T - Pval
    ax_topo = fig.add_subplot(gs[:-1,10:])
    ax_cbar = fig.add_subplot(gs[-1,12:18])
    vmin, vmax = 0.0, 0.2
    data.loc[data.t_pval>vmax]=vmax # set upper limit for pval
    mne_kwargs.update(vmin=vmin, vmax=vmax)
    ax_topo, cbar = plot_topomap(ch_names=ch_names, data=data['t_pval'], ax=ax_topo, cax=ax_cbar, 
                                 show_names=True, cmap='Reds_r', **mne_kwargs)
    cbar.ax.tick_params(labelsize=15)
    ax_topo.set_title('P-value (uncorrected)', fontsize=20)
    return fig