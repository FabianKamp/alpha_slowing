import os, glob, json
import numpy as np
import pandas as pd
import sys
sys.path.append('C:/Users/Kamp/Documents/nid/scripts')
from group_pipeline import group_pipeline
from result_visualization import plot_group_box, plot_model_fit, plot_params
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from itertools import product

class param_search():
    """
    To get the best parameter fittings for the lemon dataset we perform a grid search over some parameter settings of the FOOOF algorithm.
    1. Load meta file
    2. Set subject ids for groups (young/old)
    2. For each parameter setting: 
        + Get Parameter file
        + Create output folder and copy parameter file in there 
        + Run group pipeline on those subjects
        + Plot alpha peaks 
        + Plot model fits for both groups
        + Save figures into one pdf file
    """

    def __init__(self, subject_ids, group_folder, meta_file, out_folder):
        self.subject_ids = subject_ids
        self.group_folder = group_folder
        self.meta_file = meta_file
        self.out_folder = out_folder
        self.pdf_file = os.path.join(out_folder, f'param_results.pdf')

    def run(self, fooof_freq_range, peak_width_limits, aperiodic_mode):
        pdf = PdfPages(self.pdf_file) # open pdf file 
        n = 0
        for fq,pwl,am in product(fooof_freq_range,peak_width_limits,aperiodic_mode):
            param_out_folder = os.path.join(self.out_folder, f'params_{n:02d}')
            if not os.path.isdir(param_out_folder): 
                os.mkdir(param_out_folder)
            param_file = os.path.join(param_out_folder, f'params_{n:02d}.json')
            params = {'psd': {'window_length': 5, 'overlap': 0.5, 'window_type': 'hann'},
                      'fooof': {'freq_range': fq,
                                'aperiodic_mode': am,
                                'peak_width_limits': pwl,
                                'max_n_peaks': 6,
                                'min_peak_height': 0.1},
                      'alpha_band': [8, 13],
                      'sc_freq_range': [8, 13]}
            with open(param_file, 'w') as file: 
                json.dump(params, file)
            pipe = group_pipeline(self.group_folder, self.subject_ids, param_file, param_out_folder)
            results = pipe.run()
            self.plot_results(results, self.meta_file, params, pdf, n)
            n+=1
        pdf.close() # close pdf file
    
    def plot_results(self, results, meta_file, params, pdf, n): 
        # Dict values
        fig, ax = plt.subplots(figsize=(10,4), tight_layout=True)
        fig.suptitle(f'Parameter Settings {n:02d}')
        plot_params(params, ax)
        pdf.savefig()
        # Plot results
        # Alpha peak
        fig, ax = plt.subplots(1,2,figsize=(10,4), tight_layout=True)
        fig.suptitle('Alpha Peak Frequency')
        fig = plot_group_box(results, meta_file, 'alpha_peak_freqs', fig=fig, ax=ax[0], show_subjects=False)
        fig = plot_group_box(results, meta_file, 'alpha_peak_freqs', fig=fig, ax=ax[1], show_subjects=True, cbar=True)
        pdf.savefig()
        # Peak CF
        fig, ax = plt.subplots(1,2,figsize=(10,4), tight_layout=True)
        fig.suptitle('FOOOF - Alpha Peak CF')
        fig = plot_group_box(results, meta_file, 'alpha_peak_cf', fig=fig, ax=ax[0], show_subjects=False)
        fig = plot_group_box(results, meta_file, 'alpha_peak_cf', fig=fig, ax=ax[1], show_subjects=True, cbar=True)
        pdf.savefig()
        # Model fit
        fig, ax = plt.subplots(figsize=(5,4), tight_layout=True)
        fig.suptitle('FOOOF - Model Fit')
        plot_model_fit(results, meta_file, ax, color_outliers=True)
        pdf.savefig()