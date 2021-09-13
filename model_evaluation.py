import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from fooof.utils.io import load_fooofgroup
import pandas as pd
from scipy.signal import gaussian

class model_evaluation():
    """
    Class to evaluate the model results, fits and parameters of FOOOF. 
    Takes as input the subect-pipeline results or results.csv file and the FOOOFGroup.json object.
    Output is a pdf file with these figures
        1. Result 
        2. Model fit 
        3. Model Parameters
        4. Plot fooofed spectrum + model for models with highest parameter values
    """
    def __init__(self, subject_id, results, fooof_file, pdf_file):
        self.subject_id = subject_id
        if type(results)!=pd.DataFrame:
            results = pd.read_csv(results)
        self.results = results.loc[results.id==subject_id].reset_index()
        self.fooof_file = fooof_file
        self.fg = load_fooofgroup(file_name=fooof_file)
        self.freq_band = (8,13)
        self.pdf_file = pdf_file

    def run(self):
        with PdfPages(self.pdf_file) as pdf: 
            # First page - psds
            psds_fig = self.plot_psds(self.subject_id, self.results, self.fg, self.freq_band)
            pdf.savefig()
            # Seconde page - results
            res_fig = self.plot_results(self.subject_id, self.results, self.fg)
            pdf.savefig()
            # Third page - model fit
            eval_fig = self.plot_model_eval(self.subject_id, self.results, self.fg)
            pdf.savefig()
            # Fourth page - model params
            param_fig = self.plot_params(self.subject_id, self.results, remove_outliers=False)
            pdf.savefig()
            # Fifth page - model params without outliers
            param_fig = self.plot_params(self.subject_id, self.results, remove_outliers=True)
            pdf.savefig()
            # Last page - fooofed spectrum of models with largest params
            max_param_fig = self.plot_max_param_models(self.subject_id, self.results, self.fg)
            pdf.savefig()
    
    def plot_psds(self, subject_id, results, fg, freq_band):
        fig, ax = plt.subplots(1,2, figsize=(8,4), tight_layout=True, sharey=True)
        fig.suptitle(subject_id.capitalize(), fontsize=20)
        self.plot_raw_psds(results, fg, fig, ax[0])
        self.plot_psds_peaks(results,fg,freq_band,fig,ax[1])
        return fig
    
    def plot_raw_psds(self,results, fg, fig, ax):
        # Ploting
        norm = matplotlib.colors.Normalize(vmin=-80, vmax=80)
        cmap = matplotlib.cm.get_cmap('viridis')
        for n in range(fg.power_spectra.shape[0]):
            c = cmap(norm(results.sensor_pos_a[n]))
            ax.plot(fg.freqs, fg.power_spectra[n], color=c, alpha=0.2, linewidth=1);
        ax.grid(alpha=0.4)
        ax.set_ylabel('log(Power)')
        ax.set_yticklabels([])
        ax.set_xlabel('Frequency')
        ax.set_title('Power Spectrum')
        # Colorbar
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), alpha=0.4, fraction=0.03)
        ticks = cbar.get_ticks()[[0,-1]]
        cbar.set_ticks(ticks); 
        cbar.set_ticklabels(['Posterior','Anterior'])

    def plot_psds_peaks(self, results, fg, freq_band, fig, ax):
        min_freq, max_freq = freq_band
        # Set up colormap
        norm = matplotlib.colors.Normalize(vmin=-80, vmax=80)
        cmap = matplotlib.cm.get_cmap('viridis')

        for n, psd in enumerate(fg.power_spectra):
            # Smooth psd 
            k = np.repeat(1/3,3)
            psd = np.convolve(psd, k, 'same')
            # Plot psd
            c = cmap(norm(results.sensor_pos_a[n]))
            ax.plot(fg.freqs, psd, alpha=0.1, linewidth=1, color=c)
            # Mark the peak frequencies
            peak_freq = results.alpha_peak_freqs[n]
            if np.isnan(peak_freq):
                continue
            ax.scatter(peak_freq, psd[np.isclose(fg.freqs,peak_freq)], marker='x', alpha=0.5, s=10, color=c)
        # color freq band
        ylim = ax.get_ylim()
        ax.fill_betweenx(ylim, x1=min_freq, x2=max_freq, alpha=0.2, color='tab:blue') # Fill alpha range
        ax.set_ylim(ylim)
        ax.grid(alpha=0.4)
        ax.set_xlabel('Frequency')
        ax.set_title('Band Peak Frequencies')

    # Results
    def plot_results(self, subject_id, results, fg):     
        fig, ax = plt.subplots(1,2, figsize=(8,4), tight_layout=True)
        fig.suptitle(subject_id.capitalize() + '\nFOOOF Model Evaluation', fontsize=20)
        self.plot_alpha_peak(results, ax=ax[0])
        self.plot_peak_sensorpos(results, ax[1])

    def plot_alpha_peak(self, results, ax): 
        # Remove nan values 
        results = results.dropna(subset=['alpha_peak_cf', 'alpha_peak_freqs'])
        data = [results.alpha_peak_cf, results.alpha_peak_freqs]
        # Boxplot
        positions=[0,1]
        medianprops = dict(linewidth=2., color='tab:red')
        ax.boxplot(data, positions=positions, showcaps=False, showfliers=False, widths=0.25, medianprops=medianprops)
        # Add lines 
        for y1,y2 in zip(data[0], data[1]):
            alpha=0.08
            ax.plot(positions, [y1,y2], marker='o', color='k', alpha=alpha)
        ax.set_xticklabels(['FOOOF\nAlpha Peak CF', 'Alpha Peak Frequency'])
        ax.set_ylabel('Frequency')
        ax.set_title('Alpha Peak Frequency')

    def plot_peak_sensorpos(self, results, ax):
        alpha_peaks = results.alpha_peak_cf
        ax.scatter(results.sensor_pos_a, alpha_peaks, color='k', alpha=0.5) 
        ax.set_xlabel('Anterior-Posterior Sensor Position')
        ax.set_title('Alpha Peak vs. Sensor Position')
        for n, ch in enumerate(results.ch_names):
            text_kwargs = dict(size=5., color='tab:red', va='center', ha='center')
            ax.annotate(ch, (results.sensor_pos_a[n], alpha_peaks[n]), **text_kwargs)


    # Model Evaluation
    def plot_model_eval(self, subject_id, results, fg):
        fig, ax = plt.subplots(1,3, figsize=(12,4), tight_layout=True)
        fig.suptitle(subject_id.capitalize(), fontsize=20)
        self.plot_model_err_rsq(results, ax[0])
        self.plot_max_err(results, fg, ax[1])
        self.plot_min_rsq(results, fg, ax[2])
        return fig
    
    def plot_model_err_rsq(self, results, ax, color='k'):
        err = results.model_error
        rsq = results.model_rsquared
        # X spacing and jitter
        n = len(results)
        positions = [0.2,0.8] 
        x = lambda pos: np.random.randn(n)*0.02+pos #apply some jitter to position
        # Plot
        alpha = 0.5
        ax.scatter(x(positions[0]),err,alpha=alpha, color=color)
        ax.scatter(x(positions[1]),rsq,alpha=alpha, color=color)
        ax.set_xticks(positions)
        ax.set_yticks([0,0.1,0.2,0.5,0.8,0.9,1])
        ax.set_xticklabels(['Error','R squared'])
        ax.set_xlim([0,1])
        ax.set_title('Model Fit')
        ax.grid()

    def plot_max_err(self, results, fg, ax):
        err = np.round(np.max(results.model_error),2)
        ind = np.argmax(results.model_error)
        ch_name = results.ch_names[ind]
        ax.set_title(f'Largest Model Error: {err}\nChannel {ch_name}')
        # fm plot
        fm = fg.get_fooof(ind=ind, regenerate=True)
        fm.plot(ax=ax, add_legend=False) 

    def plot_min_rsq(self, results, fg, ax):
        rsq = np.round(np.min(results.model_rsquared),2)
        ind = np.argmin(results.model_rsquared)
        ch_name = results.ch_names[ind]
        ax.set_title(f'Smallest Model R^2: {rsq}\nChannel {ch_name}')
        # fm plot
        fm = fg.get_fooof(ind=ind, regenerate=True)
        fm.plot(ax=ax, add_legend=False) 
    
    # Parameter
    # Plot aperiodic_knee and aperiodic_exponent and aperiodic_offset
    def plot_params(self, subject_id, results, remove_outliers):
        fig, ax = plt.subplots(1,3, tight_layout=True)
        title = subject_id.capitalize() + int(remove_outliers)*' (prev. outliers removed)' 
        fig.suptitle(title, fontsize=20)
        results = results.loc[results.id==subject_id]
        params = ['model_aperiodic_offset', 'model_aperiodic_exponent', 'model_aperiodic_knee']
        
        for n, param in enumerate(params):
            label = (' '.join([w.capitalize() for w in param.split('_')][1:]))
            self.plot_param(param, results, ax=ax[n], label=label, remove_outliers=remove_outliers)
            self._despine(ax[n]) 
        return fig

    def plot_param(self, param, results, ax, label, remove_outliers=False): 
        # Remove nan values from results
        results = results.dropna(subset=[param])
        data = results[param].to_numpy()
        ch_names = results['ch_names'].to_numpy()
        # Get outliers
        idxs, outliers = self._get_outliers(data)
        # Names of the outlier channels
        outlier_names = ch_names[idxs]
        
        if remove_outliers:
            mask = np.ones(len(data))
            mask[idxs] = 0
            data = data[mask.astype('bool')]

        # Boxplot
        medianprops = dict(linewidth=2., color='tab:red')
        alpha=0.08
        ax.boxplot(data, showcaps=False, showfliers=True, widths=0.25, medianprops=medianprops);
        ax.scatter(np.repeat(1,len(data)), data, marker='o', color='k', alpha=alpha)
        ax.set_xlabel(label)
        
        if not remove_outliers:
            # Annotate outliers
            for idx, outlier in zip(idxs, outliers):
                ax.annotate(ch_names[idx], (1.1,outlier), size=6.5)
        
        return outlier_names

    def _get_outliers(self,data):
        """
        Get outlier values and indices of input array
        """
        # Define lower and upper bound
        data = np.array(data)
        upper_q = np.quantile(data,0.75)
        lower_q = np.quantile(data,0.25)
        IQR = upper_q - lower_q
        upper_bound = upper_q+1.5*IQR
        lower_bound = lower_q-1.5*IQR
        # Get outliers
        outlier_idx = np.where((data>upper_bound)|(data<lower_bound))[0]
        outlier_values = data[outlier_idx]
        return outlier_idx, outlier_values
    
    def _despine(self, ax): 
        for side in ['top', 'right', 'bottom']:
            ax.spines[side].set_visible(False) # removes top and right spine
        # truncate the y spine
        ymin, ymax = ax.get_yticks()[1], ax.get_yticks()[-2]
        ax.spines['left'].set_bounds((ymin, ymax))
        ax.set_xticks([])

    # Plot model spectra for models with largest parameter
    def plot_max_param_models(self, subject_id, results, fg):
        fig, ax = plt.subplots(1,3, figsize=(15,5),tight_layout=True)
        fig.suptitle(subject_id.capitalize(), fontsize=20)
        params = ['model_aperiodic_offset', 'model_aperiodic_exponent', 'model_aperiodic_knee']
        for n, param in enumerate(params):
            self.plot_max_param_model(param, results,fg,ax[n])
        return fig

    def plot_max_param_model(self, param, results, fg, ax):
        ind = np.argmax(results[param])
        ch_name = results.ch_names[ind]
        param_name = ' '.join([p.capitalize() for p in param.split('_')][1:])
        ax.set_title(f'Largest Model Param: {param_name}\nChannel {ch_name}')
        # fm plot
        fm = fg.get_fooof(ind=ind, regenerate=True)
        fm.plot(ax=ax, add_legend=False) 


