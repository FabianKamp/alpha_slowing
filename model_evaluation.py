import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from fooof.utils.io import load_fooofgroup

class model_evaluation():
    """
    Class to evaluate the model results, fits and parameters of FOOOF. 
    Takes as input the subect-pipeline results and the saved FOOOFGroup object.
    Output is a pdf file with these figures
        1. Result 
        2. Model fit 
        3. Model Parameters
        4. Plot fooofed spectrum + model for models with highest parameter values
    """
    def __init__(self, subject_id, results, fooof_file, pdf_file):
        self.subject_id = subject_id
        self.results = results.loc[results.id==subject_id].reset_index()
        self.fooof_file = fooof_file
        self.fg = load_fooofgroup(file_name=fooof_file)
        self.pdf_file = pdf_file

    def run(self):
        with PdfPages(self.pdf_file) as pdf: 
            # First page - results
            res_fig = self.plot_results(self.subject_id, self.results, self.fg)
            pdf.savefig()
            # Second page - model fit
            eval_fig = self.plot_model_eval(self.subject_id, self.results, self.fg)
            pdf.savefig()
            # Third page - model params
            param_fig = self.plot_params(self.subject_id, self.results, remove_outliers=False)
            pdf.savefig()
            # Fourth page - model params without outliers
            param_fig = self.plot_params(self.subject_id, self.results, remove_outliers=True)
            pdf.savefig()
            # Last page - fooofed spectrum of models with largest params
            max_param_fig = self.plot_max_param_models(self.subject_id, self.results, self.fg)
            pdf.savefig()
    
    # Results
    def plot_results(self, subject_id, results, fg):     
        fig, ax = plt.subplots(1,3, figsize=(12,4), tight_layout=True)
        fig.suptitle(subject_id.capitalize(), fontsize=20)
        self.plot_peak_sc(results, ax=ax[0], vert=True)
        self.plot_peak_sensorpos(results, ax[1])
        self.plot_psds(results, fg, fig, ax[2])

    def plot_peak_sc(self, results, ax, vert=True): 
        # Remove nan values 
        results = results.dropna(subset=['alpha_peak_cf', 'spectral_centroid'])
        data = [results.alpha_peak_cf, results.spectral_centroid]
        # Boxplot
        positions=[0,1]
        medianprops = dict(linewidth=2., color='tab:red')
        ax.boxplot(data, positions=positions, showcaps=False, showfliers=False, widths=0.25, medianprops=medianprops, vert=vert);
        # Add lines 
        for peak,sc in zip(results.alpha_peak_cf, results.spectral_centroid):
            alpha=0.08
            if vert:
                ax.plot(positions, [peak,sc], marker='o', color='k', alpha=alpha)
            else:
                ax.plot([peak,sc], positions, marker='o', color='k', alpha=alpha)
        if vert:
            ax.set_xticklabels(['Alpha Peak', 'Spectral Centroid'])
            ax.set_ylabel('Frequency')
        else:
            ax.set_yticklabels(['Alpha Peak', 'Spectral Centroid'])
            ax.set_xlabel('Frequency')
        ax.set_title('Alpha-Peak vs. Spectral Centroid')

    def plot_peak_sensorpos(self, results, ax):
        alpha_peaks = results.alpha_peak_cf
        ax.scatter(results.sensor_pos_a, alpha_peaks, color='k', alpha=0.5) 
        ax.set_xlabel('Anterior-Posterior Sensor Position')
        ax.set_title('Alpha Peak vs. Sensor Position')
        for n, ch in enumerate(results.ch_names):
            text_kwargs = dict(size=5., color='tab:red', va='center', ha='center')
            ax.annotate(ch, (results.sensor_pos_a[n], alpha_peaks[n]), **text_kwargs)
    
    def plot_psds(self,results, fg, fig, ax):
        # Ploting
        norm = matplotlib.colors.Normalize(vmin=-80, vmax=80)
        cmap = matplotlib.cm.get_cmap('viridis')
        for n in range(fg.power_spectra.shape[0]):
            c = cmap(norm(results.sensor_pos_a[n]))
            ax.plot(fg.freqs, fg.power_spectra[n], color=c, alpha=0.2, linewidth=1);
        ax.grid(alpha=0.4)
        ax.set_ylabel('log(Power)')
        ax.set_yticklabels([])
        ax.set_xlabel('Frequency');

        # colorbar
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), alpha=0.4, fraction=0.035)
        ticks = cbar.get_ticks()[[0,-1]]
        cbar.set_ticks(ticks); 
        cbar.set_ticklabels(['Posterior','Anterior'])
        
        pass


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


