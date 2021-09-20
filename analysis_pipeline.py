import os, glob, json
import numpy as np
import pandas as pd
import sys
sys.path.append('C:/Users/Kamp/Documents/nid/scripts')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import result_visualization as viz
import stat_analysis as sa
import tools 
import matplotlib.gridspec as gridspec

class analysis_pipeline():
    """
    Class to analyze and visualize the results from group_pipeline.
    1. Plot sensor summary
    2. Plot averaged subjects 
    3. Compute group statistics
    4. Compute sensor statistics
    5. Plot TopoMaps of descr and comparative sensor statistics
    """

    def __init__(self, result_file, meta_file, out_folder):
        """
        Load result + metafile and set up out folder
        """
        self.results = tools.load_results(result_file, meta_file)
        self.out_folder = out_folder

    def run(self, show_subjects=True):
        """
        Run the analysis pipeline:
        1. Plot sensor summary
        2. Plot averaged subjects 
        3. Compute group statistics
        4. Compute sensor statistics
        5. Plot TopoMaps of descr and comparative sensor statistics
        
        Creates pdf files in out_folder:
        + summary_results.pdf
        + summary_subjects.pdf
        + summary_topomaps.pdf

        Creates .csv files in out folder: 
        + group_stats.csv
        + descr_sensor_stats.csv
        + sensor_stats.csv

        :param show_subjects - bool, plot the sensor data for each subject
        """
        # Model fit
        fig, ax = plt.subplots(figsize=(5,4), tight_layout=True)
        fig.suptitle('FOOOF - Model Fit')
        viz.plot_model_fit(self.results, ax, color_outliers=True)
        pdf_file = os.path.join(self.out_folder, 'model_fit.pdf')
        with PdfPages(pdf_file) as pdf: 
            pdf.savefig(fig)

        # Sensor summary
        figs = self.plot_sensors(self.results)
        pdf_file = os.path.join(self.out_folder, 'summary_sensors.pdf')
        with PdfPages(pdf_file) as pdf: 
            for fig in figs:
                pdf.savefig(fig)

        # Subject summary
        figs = self.plot_subjects(self.results, show_subjects)
        pdf_file = os.path.join(self.out_folder, 'summary_subjects.pdf')
        with PdfPages(pdf_file) as pdf: 
            for fig in figs:
                pdf.savefig(fig)
        ## Compute group stats
        group_stats = sa.get_group_stats(self.results, ['alpha_peak_freqs', 'alpha_peak_cf', 'spectral_centroid', 'peak_centered_sc', 'centralized_sc'])
        group_stats_file = os.path.join(self.out_folder, 'group_stats.csv')
        group_stats.to_csv(group_stats_file, index=False)

        # Sensor TopoMaps
        ## Compute descr stats (mean,sd,min,max)
        descr_stats = sa.get_descr_sensor_stats(self.results, ['alpha_peak_freqs', 'alpha_peak_cf', 'spectral_centroid', 'peak_centered_sc', 'centralized_sc'])
        descr_stats_file = os.path.join(self.out_folder,'descr_sensor_stats.csv') 
        descr_stats.to_csv(descr_stats_file, index=False)
        
        ## Compute sensor stats (ttest, welch test)
        stats = sa.get_sensor_stats(self.results,['alpha_peak_freqs', 'alpha_peak_cf','spectral_centroid', 'peak_centered_sc', 'centralized_sc'])
        stats_file = os.path.join(self.out_folder,'sensor_stats.csv') 
        stats.to_csv(stats_file, index=False)

        # Plot TopoMaps
        figs = self.plot_topomaps(descr_stats, stats)
        pdf_file = os.path.join(self.out_folder, 'summary_topomaps.pdf')
        with PdfPages(pdf_file) as pdf: 
            for fig in figs:
                pdf.savefig(fig)
        plt.close('all')

    def plot_sensors(self, results):
        """
        Create overview plots that containing all sensor data
        """
        # All sensors + all subjects
        fig1, ax = plt.subplots(figsize=(10,5))
        ax.set_title('All Subjects')
        viz.plot_box(results, ['alpha_peak_freqs', 'alpha_peak_cf', 'spectral_centroid', 'peak_centered_sc'], 
                     labels=['Alpha Peak', 'Alpha CF', 'Spectral Centroid', 'Peak Centered SC'], ax=ax)
        # Split old and young Group
        fig2, ax = plt.subplots(1,2, figsize=(15,5), sharey=True)
        ax[0].set_title('Young Subjects')
        viz.plot_box(results.loc[results.age_group==1], ['alpha_peak_freqs', 'alpha_peak_cf', 'spectral_centroid', 'peak_centered_sc'], 
                    labels=['Alpha Peak', 'Alpha CF', 'Spectral Centroid', 'Peak\nCentered SC'], ax=ax[0])

        ax[1].set_title('Old Subjects')
        viz.plot_box(results.loc[results.age_group==2], ['alpha_peak_freqs', 'alpha_peak_cf', 'spectral_centroid', 'peak_centered_sc'], 
                labels=['Alpha Peak', 'Alpha CF', 'Spectral Centroid', 'Peak\nCentered SC'], ax=ax[1])
        viz.despine(ax[0])
        viz.despine(ax[1])
        plt.subplots_adjust(hspace=0)

        # Compare groups
        ## Alpha Peak
        fig3, ax = plt.subplots(1,2, figsize=(15,5), sharey=True)
        _ = viz.plot_group_box(results, 'alpha_peak_freqs', fig3, ax[0], boxwidths=0.1)
        ax[0].set_title('Alpha Peak Frequency')
        _ = viz.plot_group_box(results, 'alpha_peak_cf', fig3, ax[1], cbar=True, boxwidths=0.1)
        ax[1].set_title('Alpha Peak CF');
        plt.subplots_adjust(hspace=0)
        viz.despine(ax)

        ## Spectral centroid
        fig4, ax = plt.subplots(1,2, figsize=(15,5), sharey=True)
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

        return [fig1, fig2, fig3, fig4, fig5]
    
    def plot_subjects(self, results, show_subjects):
        """
        Plot subject sensor averages of both groups.
        """
        figs = []
        # Plot subject averages
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
            ax[1].set_title('Alpha Peak CF', pad=20);
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
            ax[1].set_title('Peak Centered SC', pad=20);
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
        return figs
    
    def plot_topomaps(self, descr_stats, stats):
        """
        Plot topomaps of descriptive and comparative sensor stats.
        """
        # Descriptive
        ## Young Group
        fig1 = viz.plot_topo_means(descr_stats, title = 'Young Group', group=1, cmap='Reds', show_names=True)
        ## Old Group
        fig2 = viz.plot_topo_means(descr_stats, title = 'Old Group', group=2, cmap='Blues', show_names=True)

        # Compare Groups
        ## Alpha peak frequency
        fig3 = viz.plot_topo_stats(stats, param='alpha_peak_freqs', title='Alpha Peak Frequency', vmin=-0.6, vmax=0.6)
        ## Alpha peak CF
        fig4 = viz.plot_topo_stats(stats, param='alpha_peak_cf', title='Alpha Peak Central Frequency', vmin=-0.8, vmax=0.8)
        ## Spectral centroid
        fig5 = viz.plot_topo_stats(stats, param='spectral_centroid', title='Spectral Centroid', vmin=-0.2,vmax=0.2)
        ## Centralized sc
        fig6 = viz.plot_topo_stats(stats, param='centralized_sc', title='Centralized Spectral Centroid', vmin=-0.08,vmax=0.08)
        return [fig1, fig2, fig3, fig4, fig5, fig6]