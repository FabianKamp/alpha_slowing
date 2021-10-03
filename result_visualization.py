import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib import cm
import statsmodels.api as sm
import mne
import sys
sys.path.append('C:/Users/Kamp/Documents/nid/scripts')
from tools import load_meta_data

# Helper Functions
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
    """
    Despines input axes.
    """
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

def plot_bar_nan(df, ax, thres=None):
    x=np.arange(len(df))
    ax.bar(x, height=df['count'], color='white',edgecolor='k')
    ax.set_xticks(x)
    ax.set_xticklabels(df.iloc[:,0], rotation=90)
    ax.grid(alpha=0.5, axis='y')
    if thres: 
        outlier_heights = df.loc[df['count']>=thres, 'count']
        x = x[df['count']>=thres]
        ax.bar(x, outlier_heights, edgecolor='firebrick', color='white')
    ax.set_ylabel('Missing Values', fontsize=15)
    ax.set_xlabel(df.columns[0].replace('_', '. ').title(), fontsize=15)

# Boxplot of alpha peaks
def plot_group_box(results, param, fig, ax, cbar=False, show_subjects=False, boxwidths=0.05):
    """
    Splits data in young and old group and makes seperate boxplots for each group.
    Plots each sensor as one data point (for each subjects ~61 points). 
    If show_subjects is True, one boxplot for each subject is plotted. Else sensors are divided into "old" and "young" group.
    """
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

def plot_model_fit(results, ax, color_outliers=False):
    """
    Scatterplot of the model error, r-squared for all subjects.
    """
    # Set up data
    data = [results.loc[results.age_group==1,'model_error'], results.loc[results.age_group==1,'model_rsquared'], 
            results.loc[results.age_group==2,'model_error'], results.loc[results.age_group==2,'model_rsquared']]
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
    ax.grid(alpha=0.3)    
    if color_outliers:
        ax.legend(bbox_to_anchor=(1,1))

def plot_params(param_dict, ax):
    """
    Print Parameter Settings into axes
    """
    s = '\n'.join([f'{k}:{val}' for k,val in param_dict.items()])
    ax.text(s=s,x=0.5,y=0.5,transform=ax.transAxes, ha='center', va='center', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_visible(False) # removes top and right spine

def plot_param_box(results, params, labels, ax,  add_violin=True, add_strips=True): 
    """
    Boxplots across all parameters, using all sensors of all subjects without taking the mean.
    """
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
        violin.set_color('grey')
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

def plot_sensor_box(results, params, add_violin=True, markers=False, annotate=False): 
    """
    For each sensor makes boxplots of the parameter values (alpha-peak-freq etc.) splitted into old and young group
    :returns list of figures
    """
    # Remove nan values and split into age groups 
    results = results.dropna(subset=params)
    young = results.loc[results.age_group==1]
    old = results.loc[results.age_group==2]
    figs = []
    for ch in sorted(results.ch_names.unique()):
        fig, ax = plt.subplots(1,len(params), figsize=(4*len(params), 4), sharey=True, tight_layout=True)
        fig.suptitle(ch, y=0.99, fontsize=20)
        for n, param in enumerate(params):           
            data = [young.loc[young.ch_names==ch, param], old.loc[old.ch_names==ch, param]]
            # Boxplot
            positions = [1,2] # young=1, old=2
            medianprops = dict(linewidth=2., color='tab:red')
            ax[n].boxplot(data, positions=positions, showcaps=False, showfliers=False, widths=0.25, medianprops=medianprops, vert=False)
            # Add violin
            if add_violin:
                vp = ax[n].violinplot(data,positions,showextrema=False, vert=False)
                for violin in vp['bodies']:
                    violin.set_alpha(0.2) 
                    violin.set_color('grey') 
            # Add lines 
            if markers:
                for d, position in zip(data,positions):
                    kwargs = dict(marker='o', color='white', edgecolor='k', linewidth=0.5, alpha=0.8, s=60)
                    ax[n].scatter(y=np.repeat(position, len(d)), x=d, **kwargs) 
            
            if annotate:
                for group in [young,old]:
                    group = group.loc[group.ch_names==ch]
                    for sub in group.id: 
                        x = group.loc[group.id==sub, param]
                        y = group.loc[group.id==sub, 'age_group']
                        ax[n].annotate(text=sub, xy=(x,y), size=5, ha='center', va='center', rotation='vertical', alpha=0.5)
            # Labeling
            ax[n].set_title(param.replace('_',' ').title(), fontsize=15)
            ax[n].set_xlabel('Frequency (Hz)', fontsize=15)
            if param=='centralized_sc': 
                ax[n].set_xlim(-0.5,0.5)
            else:
                ax[n].set_xlim(8,13)
            ax[n].grid(alpha=0.4)
        
        ax[0].set_yticks([1,2])
        ax[0].set_yticklabels(['Young', 'Old'], fontsize=15)
        figs.append(fig)
        plt.close('all')
    return figs

def plot_ecdf(results, param, ax, mean=False, label=None, percentiles=False, **scatterkwargs):
    """
    Plot the empirical cumulative distribution of the param values.
    """
    if mean: 
        data = results.groupby(['id']).mean().reset_index()[param]
    else: 
        data = results[param]
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    ax.scatter(x, y, label=label, **scatterkwargs)
    if percentiles: 
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        low, up = np.percentile(x,[25,75])
        ax.plot([low,up], [0.25,0.75], linestyle=' ', marker='o', color='k', alpha=0.6)
        linekwargs = dict(color='k', linestyle='solid', alpha=0.5, linewidth=1)
        ax.vlines(low, ymin=ylims[0], ymax=0.25, **linekwargs)
        ax.hlines(0.25, xmin=xlims[0], xmax=low, **linekwargs)
        ax.vlines(up, ymin=ylims[0], ymax=0.75, **linekwargs)
        ax.hlines(0.75, xmin=xlims[0], xmax=up, **linekwargs)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
    ax.set_ylabel('ECDF',  fontsize=15)
    ax.set_xlabel(param.replace('_',' ').title(), fontsize=15)
    ax.grid(alpha=0.5)
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
        * centralized sc
    :param data - pd.DataFrame with descr stats
    :param title - figure title
    :param group - age_group 1|2
    :param **mne_kwargs - keyword arguments that are passed to mne.viz.plot_topomap    
    :returns figure
    """
    data = data.loc[data.age_group==group].copy()
    fig = plt.figure(figsize=(20,7), tight_layout=True)
    fig.suptitle(title, fontsize=25)
    gs = gridspec.GridSpec(20, 60)
    
    # Set vmin + vmax for colors in TopoMaps
    data_mean = data.loc[data.param.isin(['alpha_peak_freqs', 'alpha_peak_cf', 'spectral_centroid']), 'mean']
    vmin, vmax = np.min(data_mean),np.max(data_mean)
    
    # Plot alpha peak frequency
    ax_topo = fig.add_subplot(gs[:-1,:15])
    ax_cbar = fig.add_subplot(gs[-1,4:11])
    p_data = data.loc[data.param=='alpha_peak_freqs', 'mean']
    ch_names = data.loc[data.param=='alpha_peak_freqs', 'ch_names']
    ax_topo, cbar = plot_topomap(ch_names=ch_names, data=p_data, ax=ax_topo, cax=ax_cbar, 
                                 vmin=vmin, vmax=vmax, **mne_kwargs)
    ax_topo.set_title('Mean Alpha\nPeak Frequency', fontsize=20)
    cbar.set_ticks([np.round(vmin,2)+0.1, np.round(vmax,2)-0.1])
    ax_cbar.tick_params(labelsize=15)
    
    # Plot alpha cf
    ax_topo = fig.add_subplot(gs[:-1,15:30])
    ax_cbar = fig.add_subplot(gs[-1,19:26])
    p_data = data.loc[data.param=='alpha_peak_cf', 'mean']
    ch_names = data.loc[data.param=='alpha_peak_cf', 'ch_names']
    ax_topo, cbar = plot_topomap(ch_names=ch_names, data=p_data, ax=ax_topo, cax=ax_cbar, 
                                 vmin=vmin, vmax=vmax, **mne_kwargs)
    ax_topo.set_title('Mean Alpha\nPeak CF', fontsize=20)
    cbar.set_ticks([np.round(vmin,2)+0.1, np.round(vmax,2)-0.1])
    ax_cbar.tick_params(labelsize=15)
    # Plot alpha sc
    ax_topo = fig.add_subplot(gs[:-1,30:45])
    ax_cbar = fig.add_subplot(gs[-1,34:41])
    p_data = data.loc[data.param=='spectral_centroid', 'mean']
    ch_names = data.loc[data.param=='spectral_centroid', 'ch_names']
    ax_topo, cbar = plot_topomap(ch_names=ch_names, data=p_data, ax=ax_topo, cax=ax_cbar, 
                                 vmin=vmin, vmax=vmax, **mne_kwargs)
    ax_topo.set_title('Mean Spectral\nCentroid', fontsize=20)
    cbar.set_ticks([np.round(vmin,2)+0.1, np.round(vmax,2)-0.1])
    ax_cbar.tick_params(labelsize=15)

    # Plot centralized sc
    ## Set vmin + vmax for colors in TopoMap
    data_mean = data.loc[data.param=='centralized_sc', 'mean']
    vmin, vmax = np.min(data_mean),np.max(data_mean)

    ax_topo = fig.add_subplot(gs[:-1,45:60])
    ax_cbar = fig.add_subplot(gs[-1,49:56])
    p_data = data.loc[data.param=='centralized_sc', 'mean']
    ch_names = data.loc[data.param=='centralized_sc', 'ch_names']
    ax_topo, cbar = plot_topomap(ch_names=ch_names, data=p_data, ax=ax_topo, cax=ax_cbar, 
                                 vmin=vmin, vmax=vmax, **mne_kwargs)
    ax_topo.set_title('Mean Centralized SC', fontsize=20)
    cbar.set_ticks([np.round(vmin,2)+0.02, np.round(vmax,2)-0.02])
    ax_cbar.tick_params(labelsize=15)
    
    plt.subplots_adjust(hspace=0)
    return fig

def plot_topo_tstats(data, param, title, mean_kwargs=dict(), t_kwargs=dict()):
    """
    Takes as input dataframe of group difference statistics. 
    Plots topomaps of the 
        * mean diff
        * t'test statistic
    :param data - pd.DataFrame with stats
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
                                 cmap='RdBu_r', **mean_kwargs)
    cbar.ax.tick_params(labelsize=15)
    ax_topo.set_title('Mean Difference', fontsize=20)
    # T - Stats
    ax_topo = fig.add_subplot(gs[:-1,10:])
    ax_cbar = fig.add_subplot(gs[-1,12:18])
    vmin, vmax = -2.5,2.5
    data.loc[data.t_pval>vmax]=vmax # set upper limit for pval
    t_kwargs.update(vmin=vmin, vmax=vmax)
    ax_topo, cbar = plot_topomap(ch_names=ch_names, data=data['t_stat'], ax=ax_topo, cax=ax_cbar, 
                                 cmap='RdBu_r', **t_kwargs)
    cbar.ax.tick_params(labelsize=15)
    ax_topo.set_title('T-Statistic', fontsize=20)
    return fig

def plot_topo_stats(stats, measure, title, **mne_kwargs):
    """
    Takes as input dataframe of sensor statistics. 
    Plots topomaps of the measure for  
        * alpha peak frequency
        * alpha peak cf
        * spectral centroid
        * centralized sc
    :param data - pd.DataFrame with sensor stats
    :param measure - statistic measure, e.g. 'spearman_corr_age'
    :param title - figure title
    :param **mne_kwargs - keyword arguments that are passed to mne.viz.plot_topomap    
    :returns figure
    """
    fig = plt.figure(figsize=(20,7), tight_layout=True)
    fig.suptitle(title, fontsize=25)
    gs = gridspec.GridSpec(20, 60)
    
    # Set vmin + vmax for colors in TopoMaps
    stats_m = stats.loc[stats.param.isin(['alpha_peak_freqs', 'alpha_peak_cf', 'spectral_centroid', 'centralized_sc']), measure]
    vmin, vmax = np.min(stats_m), np.max(stats_m)
    if ('vmin' not in list(mne_kwargs.keys())):
        mne_kwargs.update(vmin=vmin)
    if ('vmmax' not in list(mne_kwargs.keys())):
        mne_kwargs.update(vmax=vmax)
    
    # Plot topomap for each parammeter
    for n, param in enumerate(['alpha_peak_freqs', 'alpha_peak_cf', 'spectral_centroid', 'centralized_sc']):
        ax_topo = fig.add_subplot(gs[:-1,n*15:(n+1)*15]) 
        ax_cbar = fig.add_subplot(gs[-1,n*15+4:(n+1)*15-4])
        m_data = stats.loc[stats.param==param, measure]
        ch_names = stats.loc[stats.param==param, 'ch_names']
        ax_topo, cbar = plot_topomap(ch_names=ch_names, data=m_data, ax=ax_topo, cax=ax_cbar, 
                                     **mne_kwargs)
        ax_topo.set_title(' '.join(param.split('_')).title(), fontsize=20)
        #cbar.set_ticks([np.round(vmin,2)+0.1, np.round(vmax,2)-0.1])
        ax_cbar.tick_params(labelsize=15)    
    plt.subplots_adjust(hspace=0)
    return fig

# Plotting Regressions
def plot_scatter_obs(results, x_param, y_param, ax, mean=True):
    if mean:
        results = results[['id', x_param, y_param]].groupby('id').mean().reset_index()
    kwargs = dict(facecolor='white', edgecolor='black', alpha=0.5, )
    # subtracts one if plotting age group
    ax.scatter(x=results[x_param]-int(x_param=='age_group'), y=results[y_param]-int(y_param=='age_group'), **kwargs)
    ax.set_xlabel(' '.join(x_param.split('_')).title(), fontsize=15)
    ax.set_ylabel(' '.join(y_param.split('_')).title(), fontsize=15)
    ax.grid(alpha=0.5)

def plot_scatter_obs_estim(regression_res, observations, ax):
    kwargs = dict(edgecolor='k', facecolor='white', alpha=0.8)
    ax.scatter(x=regression_res.fittedvalues,y=observations, **kwargs)
    ax.text(s = f'r = {np.round(regression_res.rsquared,2)}', x=.7, y=.1, transform=ax.transAxes, fontsize=15,
            va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, pad=0.4))
    ax.grid(alpha=0.5)

def plot_uni_regression(reg_res, pred_range, ax): 
    x = np.linspace(pred_range[0], pred_range[1], 100)
    y = reg_res.predict(sm.add_constant(x))
    ax.plot(x,y, color='k', alpha=0.5)