import os, glob, json
import numpy as np
import pandas as pd
import sys
sys.path.append('C:/Users/Kamp/Documents/nid/scripts')
from subject_pipeline import subject_pipeline
from model_evaluation import model_evaluation
from pprint import pprint 
from fooof.utils.io import load_fooofgroup
import matplotlib.pyplot as plt

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
            self.data_files = [file for file, sub in zip(all_data_files, all_subject_ids) if sub in subjects]
            self.subject_ids = subjects
        
        # Output
        self.out_folder = out_folder
        self.fooof_out = os.path.join(out_folder,'fooof_objects')
        self.eval_out = os.path.join(out_folder,'model_evaluation')
        self.result_file = os.path.join(out_folder, 'all_results.csv')
    
    def run(self, evaluate=True):
        """
        Function to run the group pipeline.
        :params evaluate - creates pdf file for each participant to evaluate the model fit.
        """
        # Create outfolders
        if not os.path.isdir(self.fooof_out):
            os.mkdir(self.fooof_out)
        
        all_results = []
        for subject_id, data_file in zip(self.subject_ids, self.data_files):
            # Initialize pipeline and load data
            pipe = subject_pipeline(data_file, subject_id)
            # Run pipeline with set of parameters
            subject_results, fg = pipe.run(self.params)    
            # Save 
            fg.save(file_name=f'{subject_id}_fg.json', file_path=self.fooof_out, save_results=True, save_settings=True, save_data=True)
            all_results.append(subject_results)
            # Evaluate
            if evaluate:
                if not os.path.isdir(self.eval_out):
                    os.mkdir(self.eval_out)
                fooof_file = os.path.join(self.fooof_out,f'{subject_id}_fg.json')
                pdf_file = os.path.join(self.eval_out,f'{subject_id}_evaluation.pdf')
                evaluation = model_evaluation(subject_id, subject_results, fooof_file, pdf_file);
                evaluation.run()
        
        results_df = pd.concat(all_results)
        results_df.to_csv(self.result_file, index=False)
        return results_df
        

