import os, glob, json, sys, time
import numpy as np
import pandas as pd
sys.path.append('C:/Users/Kamp/Documents/nid/scripts')
from group_pipeline import group_pipeline
from analysis_pipeline import analysis_pipeline
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
        """
        Initialize the parameter search
        """
        self.subject_ids = subject_ids
        self.group_folder = group_folder
        self.meta_file = meta_file
        self.out_folder = out_folder
        self.pdf_file = os.path.join(out_folder, f'param_results.pdf')

    def run(self, fooof_freq_range, peak_width_limits, aperiodic_mode, alpha_band=[(8, 13)]):
        """
        Perform parameter search
        """
        n = 0
        for fq,pwl,am, ab in product(fooof_freq_range,peak_width_limits,aperiodic_mode,alpha_band):
            param_out_folder = os.path.join(self.out_folder, f'params_{n:02d}')
            if not os.path.isdir(param_out_folder): 
                os.mkdir(param_out_folder)
            # Set up params
            param_file = os.path.join(param_out_folder, f'params_{n:02d}.json')
            params = {'psd': {'window_length': 5, 'overlap': 0.5, 'window_type': 'hann'},
                        'fooof': {'freq_range': fq,
                                'aperiodic_mode': am,
                                'peak_width_limits': pwl,
                                'max_n_peaks': 6,
                                'min_peak_height': 0.1},
                        'alpha_band': ab,
                        'sc_freq_range': ab}
            with open(param_file, 'w') as file: 
                json.dump(params, file)
            # Run pipeline with param settings
            start_time = time.time() # time each lapse
            # Group pipeline
            group_pipe = group_pipeline(self.group_folder, self.subject_ids, param_file, param_out_folder)
            _ = group_pipe.run()
            # Analysis
            analysis_out = os.path.join(param_out_folder, 'analysis')
            if not os.path.isdir(analysis_out): 
                os.mkdir(analysis_out)
            analysis_pipe = analysis_pipeline(group_pipe.result_file, self.meta_file, analysis_out)
            analysis_pipe.run()
            print(f'{n:02d} parameter setting finished. Time: {np.round(time.time()-start_time)}s')
            n+=1