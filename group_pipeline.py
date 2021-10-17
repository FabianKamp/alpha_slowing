import os, glob, json
import numpy as np
import pandas as pd
import sys
sys.path.append('C:/Users/Kamp/Documents/nid/scripts')
from subject_pipeline import subject_pipeline
from model_evaluation import model_evaluation
from fooof.utils.io import load_fooofgroup
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages

class group_pipeline():
    """
    Takes as input a set of parameters (defined in .json file) and a folder containing eeg data from each subject.
    Iterate over participants in group and for each subject:

    1. Apply subject pipeline 
        * Input parameters 
        * Save FOOOFobject 
        * Concatenate result files
    2. Apply model evaluation
        * Input results + FOOOFobject file
        * Create PDF with results + model fits and parameters for each subject
        * Create PDF with all psds for all subjects

    Saves result in .csv file with data from all participants.
    """
    def __init__(self, group_folder, subjects, param_file, out_folder): 
        """
        :params group_folder - folder with subject eeg data folders
        :subjects - 'all', int or list defines which/how many subjects of the group are processed
        :param_file - .json file with the parameters for the subject pipeline, e.g. fooof parameters etc.
        :outfolder - folder to save the output  
        """
        # Load Parameters
        self.param_file = param_file
        with open(self.param_file) as param_file:
            self.params = json.load(param_file)
        
        # Get data files
        self.group_folder = group_folder
        all_data_files = glob.glob(self.group_folder + '**\\*EC.set')
        all_subject_ids = [file.split('\\')[-2] for file in all_data_files]
        if subjects=='all':
            self.data_files=all_data_files
            self.subject_ids=all_subject_ids
        elif type(subjects)==int:
            self.data_files=all_data_files[:subjects]
            self.subject_ids=all_subject_ids[:subjects]
        elif type(subjects)==list:
            subject_files = [(sub, file) for file, sub in zip(all_data_files, all_subject_ids) if sub in subjects]
            self.subject_ids, self.data_files = zip(*subject_files)
        
        # Output
        self.out_folder = out_folder
        self.fooof_out = os.path.join(out_folder,'fooof_objects')
        self.eval_out = os.path.join(out_folder,'model_evaluation')
        self.result_file = os.path.join(out_folder, 'all_results.csv')
    
    def run(self, evaluate=True, n_jobs=3):
        """
        Function to run the group pipeline.
        :params evaluate - creates pdf file for each participant to evaluate the model fit.
        """
        # Create outfolders
        if not os.path.isdir(self.fooof_out):
            os.mkdir(self.fooof_out)
        # Run subject pipelines in parallel        
        results = Parallel(n_jobs=n_jobs)(delayed(self._run_subject_pipe)(subject_id, data_file) 
                                          for subject_id, data_file in zip(self.subject_ids, self.data_files))
        results_df = pd.concat(results)
        # Save results to file
        results_df.to_csv(self.result_file, index=False)

        # Run model evaluation in parallel
        if evaluate:
            if not os.path.isdir(self.eval_out):
                os.mkdir(self.eval_out)
            Parallel(n_jobs=n_jobs)(delayed(self._run_subject_eval)(subject_id, results_df)  
                                                for subject_id in self.subject_ids)   
            #Save psds figs into one file
            #psds_file = os.path.join(self.eval_out, 'all_psds.pdf')
            #with PdfPages(psds_file) as file:
            #    for fig in psds_figs: 
            #        file.savefig(fig)

        return results_df
    
    def _run_subject_pipe(self, subject_id, data_file):
        """
        Function to run the subject pipeline for each subject
        """
        # Initialize pipeline and load data
        pipe = subject_pipeline(data_file, subject_id)
        # Run pipeline with set of parameters
        subject_results, fg = pipe.run(self.params)    
        # Save 
        fooof_file = os.path.join(self.fooof_out,f'{subject_id}_fg.json')
        fg.save(file_name=fooof_file, save_results=True, save_settings=True, save_data=True)
        return subject_results
        
    def _run_subject_eval(self, subject_id, results):
        # Evaluate
        freq_band = self.params['alpha_band']
        fooof_file = os.path.join(self.fooof_out,f'{subject_id}_fg.json')
        pdf_file = os.path.join(self.eval_out,f'{subject_id}_evaluation.pdf')
        evaluation = model_evaluation(subject_id, results, fooof_file, freq_band, pdf_file)
        evaluation.run()
        #psds_fig = evaluation.plot_psds(evaluation.subject_id, evaluation.results, eval.freq_band, self.fg)
        #return psds_fig


