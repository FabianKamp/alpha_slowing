import os, glob, json
import numpy as np
import pandas as pd
import sys
sys.path.append('C:/Users/Kamp/Documents/nid/scripts')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import result_visualization as viz
import stat_analysis as sa
import tools 


class analysis_pipeline():
    """
    Class to analyze and visualize the results from group_pipeline.
        * Count missing values for each parameter
        * Plot sensor summary
        * Plot averaged subjects 
        * Compute group statistics
        * Compute sensor statistics
        * Compute permutation cluster test
        * Plot TopoMaps of descr and comparative sensor statistics
        * Plot boxplot of all sensors' parameter values
    """

    def __init__(self, result_file, meta_file, out_folder, age_file=None, name_match_file=None):
        """
        Load result + metafile + exact ages and set up out folder
        """
        self.results = tools.load_results(result_file, meta_file, age_file, name_match_file)
        self.out_folder = out_folder

    def run(self, params=None, show_subjects=False):
        """
        Run the analysis pipeline:
        * Count missing values for each parameter
        * Plot sensor summary
        * Plot averaged subjects 
        * Compute group statistics
        * Compute sensor statistics
        * Compute permutation cluster test
        * Plot TopoMaps of descr and comparative sensor statistics
        * Plot boxplot of all sensors' parameter values
        
        Creates pdf files in out_folder:
        + na_counts.pdf
        + summary_sensors.pdf
        + summary_subjects.pdf
        + summary_topomaps.pdf
        + all_sensors.pdf

        Creates .csv files in out folder: 
        + group_stats.csv
        + descr_sensor_stats.csv
        + sensor_stats.csv

        :param params, columns of result data frame that are used for the anaylsis
        :param show_subjects - bool, plot the sensor data for each subject
        """
        if params is None: 
            params = ['alpha_peak_freqs', 'alpha_peak_cf', 'spectral_centroid', 'peak_centered_sc', 'centralized_sc']
        # Check if all params are in dataframe
        assert np.all([p in self.results.columns for p in params]), "Params must be columns of the results data frame."

        # Missing values count
        figs = self.plot_count_na(self.results, ['ch_names'] + params, 
                                  thres=10)
        pdf_file = os.path.join(self.out_folder, 'na_counts.pdf')
        self.save_figures(pdf_file, figs)                     

        # Model fit
        fig, ax = plt.subplots(figsize=(5,4), tight_layout=True)
        fig.suptitle('FOOOF - Model Fit')
        viz.plot_model_fit(self.results, ax, color_outliers=True)
        pdf_file = os.path.join(self.out_folder, 'model_fit.pdf')
        self.save_figures(pdf_file, fig)

        # Sensor summary
        figs = self.plot_sensors(self.results)
        pdf_file = os.path.join(self.out_folder, 'summary_sensors.pdf')
        self.save_figures(pdf_file, figs)

        # Subject summary
        figs = self.plot_subjects(self.results, show_subjects)
        pdf_file = os.path.join(self.out_folder, 'summary_subjects.pdf')
        self.save_figures(pdf_file, figs)
        
        ## Compute group stats
        group_stats = sa.get_group_stats(self.results, params)
        group_stats_file = os.path.join(self.out_folder, 'group_stats.csv')
        group_stats.to_csv(group_stats_file, index=False)

        # Sensor TopoMaps
        ## Compute descr stats (mean,sd,min,max)
        descr_stats = sa.get_descr_sensor_stats(self.results, params)
        descr_stats_file = os.path.join(self.out_folder,'descr_sensor_stats.csv') 
        descr_stats.to_csv(descr_stats_file, index=False)
        
        ## Compute sensor stats (ttest, welch test)
        stats = sa.get_sensor_stats(self.results, params)
        stats_file = os.path.join(self.out_folder,'sensor_stats.csv') 
        stats.to_csv(stats_file, index=False)

        ## Compute permutation cluster test
        sensor_clusters = sa.permutation_cluster_test(self.results, params=params, 
                                                      thres=2)
        cluster_file = os.path.join(self.out_folder, 'sensor_clusters.csv')
        sensor_clusters.to_csv(cluster_file, index=False)

        # Plot TopoMaps
        figs = self.plot_stat_topomaps(descr_stats, stats, sensor_clusters)
        pdf_file = os.path.join(self.out_folder, 'summary_topomaps.pdf')
        self.save_figures(pdf_file, figs)
        
        # Plot all sensors' parameter values
        figs = viz.plot_sensor_box(self.results, params, markers=True)
        pdf_file = os.path.join(self.out_folder, 'all_sensors.pdf')
        self.save_figures(pdf_file, figs)

        # Regression
        ## Univariate regressions
        uni_lin_reg = sa.univariate_lin_regression(self.results,params) # returns dictionary {param:reg_result} 
        uni_log_reg = sa.univariate_log_regression(self.results,params)
        ## Save as text file
        reg_folder = os.path.join(self.out_folder, 'regression')
        if not os.path.isdir(reg_folder):
            os.mkdir(reg_folder)
        for label, uni_reg in zip(['lin', 'log'], [uni_lin_reg, uni_log_reg]):
            for param, res in uni_reg.items():
                file_path = os.path.join(reg_folder, f'{param}_univariate_{label}_regression.txt')
                with open(file_path, 'w') as file: 
                    file.write(res.summary().as_text())
        ## Plot
        lin_fig = self.plot_uni_regression(self.results, uni_lin_reg, label='linear')
        log_fig = self.plot_uni_regression(self.results, uni_log_reg, label='logistic')
        pdf_file = os.path.join(self.out_folder, 'univariate_regressions.pdf')
        self.save_figures(pdf_file, figs=[lin_fig, log_fig])

        ## Multivariate regressions
        multi_lin_reg = sa.multivariate_lin_regression(self.results, params)
        ## Save as text/csv files
        for param, (res, coeff) in multi_lin_reg.items():
            file_path = os.path.join(reg_folder, f'{param}_multivariate_lin_regression.txt')
            with open(file_path, 'w') as file: 
                file.write(res.summary().as_text())
            coeff.to_csv(os.path.join(reg_folder, f'{param}_multivariate_lin_regression_coeff.csv'), index=False)
        
        # Plot 
        multi_lin_figs = self.plot_multi_regression(self.results, multi_lin_reg, label='linear')
        pdf_file = os.path.join(self.out_folder, 'mulivariate_regressions.pdf')
        self.save_figures(pdf_file, multi_lin_figs)

    def plot_count_na(self, results, params, thres=None): 
        figs = []
        for param in params:
            fig, axes=plt.subplots(2,1, figsize=(10,8), tight_layout=True)
            fig.suptitle(f"Parameter: {param.replace('_', ' ').title()}", fontsize=15)
            ids_count, chs_count = tools.count_missing_param_values(results, param=param)
            for df, ax in zip([ids_count, chs_count], axes):
                viz.plot_bar_nan(df, ax, thres=thres)
            figs.append(fig)
            plt.close(fig)
        return figs
        
    def plot_sensors(self, results):
        """
        Create overview plots that containing all sensor data
        """
        # All sensors + all subjects
        fig1, ax = plt.subplots(figsize=(10,5))
        ax.set_title('All Subjects')
        viz.plot_param_box(results, ['alpha_peak_freqs', 'alpha_peak_cf', 'spectral_centroid', 'peak_centered_sc'], 
                     labels=['Alpha Peak', 'Alpha CF', 'Spectral Centroid', 'Peak Centered SC'], ax=ax)
        # Split old and young Group
        fig2, ax = plt.subplots(1,2, figsize=(15,5), sharey=True)
        ax[0].set_title('Young Subjects')
        viz.plot_param_box(results.loc[results.age_group==1], ['alpha_peak_freqs', 'alpha_peak_cf', 'spectral_centroid', 'peak_centered_sc'], 
                    labels=['Alpha Peak', 'Alpha CF', 'Spectral Centroid', 'Peak\nCentered SC'], ax=ax[0])

        ax[1].set_title('Old Subjects')
        viz.plot_param_box(results.loc[results.age_group==2], ['alpha_peak_freqs', 'alpha_peak_cf', 'spectral_centroid', 'peak_centered_sc'], 
                labels=['Alpha Peak', 'Alpha CF', 'Spectral Centroid', 'Peak\nCentered SC'], ax=ax[1])
        viz.despine(ax[0])
        viz.despine(ax[1])
        plt.subplots_adjust(hspace=0)

        # Compare groups
        ## Alpha Peak
        fig3, ax = plt.subplots(1,2, figsize=(10,5), sharey=True)
        _ = viz.plot_group_box(results, 'alpha_peak_freqs', fig3, ax[0], boxwidths=0.1)
        ax[0].set_title('Alpha Peak Frequency')
        _ = viz.plot_group_box(results, 'alpha_peak_cf', fig3, ax[1], cbar=True, boxwidths=0.1)
        ax[1].set_title('Alpha Peak CF');
        plt.subplots_adjust(hspace=0)
        viz.despine(ax)

        ## Spectral centroid
        fig4, ax = plt.subplots(1,2, figsize=(10,5), sharey=True)
        _ = viz.plot_group_box(results, 'spectral_centroid', fig4, ax[0], boxwidths=0.1 )
        ax[0].set_title('Spectral Centroid')
        _ = viz.plot_group_box(results, 'peak_centered_sc', fig4, ax[1], cbar=True, boxwidths=0.1)
        ax[1].set_title('Peak Centered SC');
        plt.subplots_adjust(hspace=0)
        viz.despine(ax)

        ## Centralized SC
        fig5, ax = plt.subplots(figsize=(5,5), sharey=True)
        _ = viz.plot_group_box(results, 'centralized_sc', fig4, ax, boxwidths=0.3 )
        ax.set_title('Centralized Spectral Centroid')
        viz.despine(ax)

        plt.close('all')
        return [fig1, fig2, fig3, fig4, fig5]
    
    def plot_subjects(self, results, show_subjects):
        """
        Plot subject sensor averages of both groups.
        """
        figs = []
        # ECDF 
        fig, ax = plt.subplots(1,4, figsize=(16,4), sharey=True, tight_layout=True)
        for n, param in enumerate(['alpha_peak_freqs', 'alpha_peak_cf', 'spectral_centroid', 'centralized_sc']):
            kwargs = dict(facecolor='white', alpha=0.7)
            viz.plot_ecdf(results.loc[results.age_group==1], param=param, mean=True, label='Young', ax=ax[n], percentiles=True, edgecolor='firebrick', **kwargs)
            viz.plot_ecdf(results.loc[results.age_group==2], param=param, mean=True, label='Old', ax=ax[n], percentiles=True, edgecolor='k', **kwargs)
        ax[n].legend(bbox_to_anchor=(1.,1.), loc='upper left', fontsize='large', markerscale=1.5)
        figs.append(fig)

        # Box-plots of subject averages
        ## Alpha peak
        boxkwargs = dict(showmeans=True, meanline=True, meanprops=dict(color='k'))
        fig, axes = plt.subplots(1,2,figsize=(10,5), sharey=True, tight_layout=True)
        _ = viz.plot_sub_means(results, 'alpha_peak_freqs', title='Alpha Peak Frequency', ax=axes[0], annotate=False, **boxkwargs)
        _ = viz.plot_sub_means(results, 'alpha_peak_cf', title='Alpha Peak Central Frequency', ax=axes[1], annotate=False, **boxkwargs)
        viz.despine(axes)
        figs.append(fig)

        ## Spectral centroid
        fig, axes = plt.subplots(1,2,figsize=(10,5),  sharey=True, tight_layout=True)
        _ = viz.plot_sub_means(results, 'spectral_centroid', title='Spectral Centroid', ax=axes[0], annotate=False, **boxkwargs)
        _ = viz.plot_sub_means(results, 'peak_centered_sc', title='Peak Centered Spectral Centroid', ax=axes[1], annotate=False, **boxkwargs)
        viz.despine(axes)
        figs.append(fig)

        # Centralized SC
        ## Subject averages and all subjects sensors
        fig,ax = plt.subplots(figsize=(5,5),  tight_layout=True)
        _ = viz.plot_sub_means(results, 'centralized_sc', title='Centralized Spectral Centroid', ax=ax, annotate=False, **boxkwargs)
        plt.subplots_adjust(hspace=0)
        viz.despine(ax)
        figs.append(fig)

        if show_subjects:
            # Show sensor data for each subject
            # Alpha Peak
            fig, ax = plt.subplots(1,2, figsize=(15,5), sharey=True, tight_layout=True)
            _ = viz.plot_group_box(results, 'alpha_peak_freqs', fig, ax[0], show_subjects=True, boxwidths=0.04)
            ax[0].set_title('Alpha Peak Frequency', pad=20)
            ax[0].tick_params(axis='x', labelsize=10)
            _ = viz.plot_group_box(results, 'alpha_peak_cf', fig, ax[1], cbar=True, show_subjects=True, boxwidths=0.04)
            ax[1].set_title('Alpha Peak CF', pad=20)
            ax[1].tick_params(axis='x', labelsize=10)
            plt.subplots_adjust(hspace=0)
            viz.despine(ax)
            figs.append(fig)

            # Spectral Centroid
            fig, ax = plt.subplots(1,2, figsize=(15,5), sharey=True, tight_layout=True)
            _ = viz.plot_group_box(results, 'spectral_centroid', fig, ax[0], show_subjects=True, boxwidths=0.04)
            ax[0].set_title('Spectral Centroid', pad=20)
            ax[0].tick_params(axis='x', labelsize=10)
            _ = viz.plot_group_box(results, 'peak_centered_sc', fig, ax[1], cbar=True, show_subjects=True, boxwidths=0.04)
            ax[1].set_title('Peak Centered SC', pad=20)
            ax[1].tick_params(axis='x', labelsize=10)
            plt.subplots_adjust(hspace=0)
            viz.despine(ax)
            figs.append(fig)

            # Centralized SC
            fig,ax = plt.subplots(figsize=(8,5),  tight_layout=True)
            _ = viz.plot_group_box(results, 'centralized_sc', fig, ax, cbar=True, show_subjects=True, boxwidths=0.04)
            ax.set_title('Centralized Spectral Centroid', pad=20)
            ax.tick_params(axis='x', labelsize=10)
            viz.despine(ax)
            figs.append(fig)
        plt.close('all')
        return figs
    
    def plot_stat_topomaps(self, descr_stats, stats, sensor_clusters):
        """
        Plot topomaps of descriptive and comparative sensor stats.
        """
        figs = []
        # Descriptive
        ## Young Group
        fig1 = viz.plot_topo_means(descr_stats, title = 'Young Group', group=1, cmap='Reds', show_names=True)
        figs.append(fig1)
        ## Old Group
        fig2 = viz.plot_topo_means(descr_stats, title = 'Old Group', group=2, cmap='Blues', show_names=True)
        figs.append(fig2)
        
        # Compare Groups
        params = ['alpha_peak_freqs', 'alpha_peak_cf', 'spectral_centroid', 'centralized_sc']
        ## Controls aspects of mean difference plot
        mean_kwargs = [dict(vmin=-0.6, vmax=0.6,show_names=True), 
                       dict(vmin=-0.8, vmax=0.8,show_names=True), 
                       dict(vmin=-0.2,vmax=0.2,show_names=True), 
                       dict(vmin=-0.08,vmax=0.08,show_names=True)]
        # Iterate over parameter
        for param, kwargs in zip(params, mean_kwargs):
            assert np.all(sensor_clusters.loc[sensor_clusters.param==param,'ch_names']==\
                    stats.loc[stats.param==param,'ch_names']),\
                    'Channel Names must be equal.'
            # Mask of sensors in cluster
            mask = sensor_clusters.loc[sensor_clusters.param==param,'mask_01'].fillna(False)
            t_kwargs = dict(mask=mask, mask_params=dict(markersize=18), show_names=True)
            # Plot topomap
            fig = viz.plot_topo_tstats(stats, param=param, title=' '.join(param.split('_')).title(), 
                                       mean_kwargs=kwargs, t_kwargs=t_kwargs)
            # Add Pvalue to plot
            pval = np.mean(sensor_clusters.loc[sensor_clusters.param==param,'pval_01'])
            if not np.isnan(pval):
                fig.text(s=f'p-val={np.round(pval,2)}',x=0.85,y=0.8, fontsize = 15, bbox=dict(boxstyle='round', alpha=0.5, facecolor='white', edgecolor='k'))
            # Append to figures
            figs.append(fig)
        
        # Plot mass bivariate correlations
        # Age Years
        fig7 = viz.plot_topo_stats(stats, 'pearsonr_age_years', 'Pearson Correlation with Age Years', vmin=-0.4, vmax=0.4)
        figs.append(fig7)
        # Age Group
        fig8 = viz.plot_topo_stats(stats, 'spearmanr_age_group', 'Spearman Correlation with Age Group', vmin=-0.4, vmax=0.4)
        figs.append(fig8)
        # Age Bin
        fig9 = viz.plot_topo_stats(stats, 'spearmanr_age_bin', 'Spearman Correlation with Age Bin', vmin=-0.4, vmax=0.4)
        figs.append(fig9)
        plt.close('all')
        return figs
    
    def plot_uni_regression(self, results, reg_results, label): 
        """
        :param reg_results, dictionary of regression results, e.g. {'alpha_peak_freq':reg_result}
        """
        # Set up dependent var and prediction range for logistic/linear univariate regression
        settings = {'linear':('age_years',[8,12]), 'logistic':('age_group',[4,16])}
        fig, ax = plt.subplots(1, len(reg_results), figsize=(len(reg_results)*4,4), tight_layout=True)
        if len(reg_results)==1: ax=[ax]
        for n, (param, reg_res) in enumerate(reg_results.items()): 
            dependent_var, pred_range = settings[label]
            if param=='centralized_sc': pred_range = [-0.5,0.5]
            viz.plot_scatter_obs(results, x_param=param, y_param=dependent_var, ax=ax[n])
            viz.plot_uni_regression(reg_res, pred_range, ax[n])
            ax[n].set_title(f'Univariate {label.title()} Regression', fontsize=15, pad=20)
        plt.close('all')
        return fig
    
    def plot_multi_regression(self, results, reg_results, label):
        n = len(reg_results)
        settings = {'linear':'age_years', 'logistic':'age_group'}
        dependent_var = settings[label]
        # Topomap
        fig_topo = plt.figure(figsize=(n*5, 5), tight_layout=True)
        fig_topo.suptitle('Coefficients\nMultivariate Linear Regression', fontsize=15)
        gs = matplotlib.gridspec.GridSpec(10,n*5)
        # Scatter
        fig_scatter, ax_sc = plt.subplots(1,n, figsize=(n*5, 5), tight_layout=True)
        if n==1: ax_sc = [ax_sc]
        fig_scatter.suptitle('True vs. Estimated Values', fontsize=15)
        for n, (param, values) in enumerate(reg_results.items()):
            reg_result, coeff = values 
            # Topo
            ax, cax = fig_topo.add_subplot(gs[:-1,n*5:(n+1)*5]), fig_topo.add_subplot(gs[-1,n*5+1:(n+1)*5-1])
            viz.plot_topomap(coeff.name[1:], coeff.coeff[1:], ax=ax, cax=cax) # first coeff is intercept
            ax.set_title(param.replace('_',' ').title(), fontsize=15)
            # Scatter
            _, _, _, endog = sa.get_exog_endog(results, param, dependent_var=dependent_var) # get endog data used for regression
            viz.plot_scatter_obs_estim(reg_result, endog, ax_sc[n])
            ax_sc[n].set_ylabel('True Age', fontsize=15)
            ax_sc[n].set_xlabel('Estimated Age', fontsize=15)
            ax_sc[n].set_title(param.replace('_',' ').title(), fontsize=15)
        plt.close('all')
        return [fig_topo, fig_scatter]

    def save_figures(self, pdf_file, figs):
        # If only one figure is passed
        if type(figs)!=list: 
            figs = [figs]
        with PdfPages(pdf_file) as pdf: 
            for fig in figs:
                pdf.savefig(fig) 

