import mne
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, gaussian
from fooof import FOOOFGroup
from fooof.analysis.periodic import get_band_peak

class subject_pipeline():
    """
    Class to extract alpha peak/spectral centroid for subject. 
    Steps of the pipeline:
        1. Calculate the PSD for each channel. Params:
            * Window length
            * Window overlap
            * Window type
        2. Fit FOOOF. Params:
            * Frequency range
            * peak_width_limits, max_n_peaks, peak_thresh
            * aperiodic_mode
        3. Extract Alpha-Band Peak. Params:
            * Alpha range
            * Only largest peaks if there is more than one peak in the alpha band
        4. Spectral centroid. Paramss
            * Frequency range
    results: pd.Dataframe with 
        alpha-peak parameters, 
        aperiodic model parameters, 
        model error/rsquared and 
        the spectral centroid
    """
    def __init__(self, data_file, subject_id):
        # Set up class and load data
        self.id = subject_id
        self.data_file = data_file
        assert str(self.id) in str(self.data_file), 'Data filename does not contain subject number.'
        self.raw = self.load_data(data_file)

    
    def load_data(self, data_file):
        """
        Load data file using mne.io.read_raw_eeglab.
        :params datafile path
        :return mne.raw 
        """
        raw = mne.io.read_raw_eeglab(data_file, verbose=False)
        data,time=raw[:]
        # Print some info
        sfreq = raw.info['sfreq']
        nchan = raw.info['nchan']
        duration = raw.times[-1]
        report = f"{data_file.split('/')[-1]} loaded.\nSampling Frequency:\t{sfreq}Hz\nNumber of Channels:\t{nchan}\nTotal length:\t\t{duration}s"
        print(report)
        return raw
    
    def run(self, params):
        """
        Run analysis pipeline.
        """
        self.params = params
        # PSD
        freqs, psds = self.get_psd(self.raw, **params['psd'])
        # Alpha Peak Freq 
        alpha_peak_freqs = self.get_peak_freqs(freqs, psds, self.params['alpha_band'])
        # FOOOF
        fg = self.fit_fooof(freqs,psds,**params['fooof'])
        fg_error = fg.get_params('error')
        fg_rsq = fg.get_params('r_squared')
        fg_offset = fg.get_params('aperiodic_params', 'offset')
        fg_exp = fg.get_params('aperiodic_params', 'exponent')
        fg_knee = fg.get_params('aperiodic_params', 'knee')
        ## Alpha Peak
        highest_peaks, all_peaks, peak_counts = self.get_band_peaks(fg, params['alpha_band'])
        # Spectral Centroid
        scs = self.get_sc(freqs, psds, freq_range=params['sc_freq_range'])
        # Peak Centered SC
        p_scs, centralized_scs = self.get_peak_centered_sc(freqs, psds, alpha_peak_freqs, width=3)

        # Get Sensor positions
        r,a,s = self._get_sensor_pos(self.raw)
        # Output
        results = {
            'id':self.id,
            'ch_names':self.raw.info['ch_names'], 
            'ch_idx':np.arange(self.raw.info['nchan']),
            'alpha_peak_freqs':alpha_peak_freqs,
            'alpha_peak_cf':highest_peaks[:,0],
            'alpha_peak_pw':highest_peaks[:,1],
            'alpha_peak_bw':highest_peaks[:,2],
            'alpha_peak_count':peak_counts,
            'model_error':fg_error,
            'model_rsquared':fg_rsq, 
            'model_aperiodic_offset':fg_exp,
            'model_aperiodic_exponent':fg_exp,
            'model_aperiodic_knee':fg_knee,
            'spectral_centroid':scs,
            'peak_centered_sc': p_scs,
            'centralized_sc': centralized_scs,
            'sensor_pos_r': r,
            'sensor_pos_a': a,
            'sensor_pos_s': s
        }
        
        # Check if the result arrays have the same length
        result_len = {k:len(val) for k, val in results.items()}
        assert len(set(result_len.values()))==2, f'{self.id}: Not all results have the same length. {result_len}'
        self.results = pd.DataFrame(results)
        return self.results, fg
    
    def get_psd(self, raw, window_length, overlap, window_type):
        """
        Calculates PSD for each channel
        """
        sfreq = raw.info['sfreq']
        n_fft = int(window_length*sfreq)
        n_overlap = n_fft*overlap
        psds, freqs = mne.time_frequency.psd_welch(raw, n_fft=n_fft, n_overlap=n_overlap, window=window_type, average='mean')
        return freqs, psds

    def get_peak_freqs(self, freqs, psds, freq_range, smooth=True):
        """
        Return the frequency with maximal power within freq range.
        """
        log_psds = np.log10(psds) 
        # Freq mask
        freq_mask = (freqs>=freq_range[0])&(freqs<=freq_range[1]) 
        m_freqs = freqs[freq_mask]
        # Iterate over channels and get frequency with max log(power)
        peak_freqs = []
        for ch in range(log_psds.shape[0]):
            psd = log_psds[ch]
            if smooth: 
                kernel = gaussian(5,3)
                psd = np.convolve(psd, kernel, 'same')
            # Mask for the freq range
            m_psd = psd[freq_mask]
            peak_idxs = find_peaks(m_psd)[0]
            # if no peak is found append nan
            if len(peak_idxs)==0:
                peak_freqs.append(np.nan)
                continue
            max_peak = np.max(m_psd[peak_idxs])
            peak_freqs.append(m_freqs[m_psd==max_peak][0])
        return peak_freqs
    
    def set_freq_range(self,freqs, psds, freq_range):
        """
        Reduce frequency range of psd.
        """
        mask = (freqs>=freq_range[0])&(freqs<=freq_range[1])
        psds, freqs = psds[:,mask]*10**12, freqs[mask] #Power converted to microVolt
        return freqs, psds

    def fit_fooof(self, freqs, psds, freq_range, aperiodic_mode, peak_width_limits, max_n_peaks, min_peak_height):
        """
        Fit FOOOF to the psd of each channel. Using FOOOFGroup object
        :returns: FOOOFGroup object, freqs and psds 
        """ 
        # Set frequency range for model       
        freqs, psds = self.set_freq_range(freqs, psds, freq_range)
        # Fit fooof for all channels
        fg = FOOOFGroup(aperiodic_mode=aperiodic_mode, peak_width_limits=peak_width_limits, max_n_peaks=max_n_peaks, min_peak_height=min_peak_height)
        fg.fit(freqs, psds)
        return fg
    
    def get_band_peaks(self, fg, alpha_band): 
        """
        :returns max_band_peaks: params of heightest peaks in alpha frequency range
        :returns band_peaks: params of peaks in alpha frequency range. 
        :returns peak_counts: count of peaks in alpha frequency range for each model.
        """
        highest_peaks = np.empty((0,3))
        all_peaks = np.empty((0,4))
        # Iterate over all models in fooof-group
        for n, fm in enumerate(fg):
            # Get highest band peak + add model number
            highest_peak = get_band_peak(fm.peak_params, alpha_band, select_highest=True)
            highest_peaks = np.vstack((highest_peaks,highest_peak))
            # Get all band peaks
            fm_all_peaks = get_band_peak(fm.peak_params, alpha_band, select_highest=False)
            # Add model numbers
            if len(fm_all_peaks.shape)>1:
                n = np.ones((fm_all_peaks.shape[0],1))*n
            fm_all_peaks = np.hstack((fm_all_peaks,n))
            all_peaks = np.vstack((all_peaks, fm_all_peaks))
        # Count number of peaks in frequency band for each model
        models = np.unique(all_peaks[:,-1])
        peak_counts = np.array([np.sum(all_peaks[:,-1]==m) for m in models])
        return highest_peaks, all_peaks, peak_counts

    def get_sc(self, freqs, psds, freq_range):
        """
        Calculates the spectral centroid from the input PSD in the frequency range.
        :params: Frequencies
        :params: PSD
        :param: Frequency range of interest
        :returns: Spectral Centroid
        """ 
        # Set frequency range
        min_freq, max_freq = freq_range
        freq_mask = (freqs>min_freq)&(freqs<max_freq)
        freqs,psds = freqs[freq_mask], psds[:,freq_mask]
        # Compute spectral centroid for channels
        scs = []
        for n in range(psds.shape[0]):
            psd = psds[n]
            sc = np.dot(freqs,psd)/np.sum(psd)
            scs.append(sc)
        scs = np.array(scs)    
        return scs

    def get_peak_centered_sc(self, freqs, psds, peak_freqs, width):
        """
        Calculates the spectral centroid of the psds in window around the peak frequency.
        :params: Frequencies
        :params: PSD
        :param: Peak frequency
        :param: width - width of the frequency window
        :returns: peak centered spectral centroid
        """
        peak_centered_scs = []
        centralized_scs = [] 
        for n, peak_freq in enumerate(peak_freqs):
            min_freq, max_freq = peak_freq-(width/2), peak_freq+(width/2)
            freq_mask = (freqs>min_freq)&(freqs<max_freq)
            # Mask freqs and psds
            m_psd = psds[n, freq_mask]
            m_freqs = freqs[freq_mask]
            # Compute sc
            peak_centered_sc = np.dot(m_freqs,m_psd)/np.sum(m_psd)
            peak_centered_scs.append(peak_centered_sc)
            # Centralize 
            centralized_sc = peak_centered_sc - peak_freq
            centralized_scs.append(centralized_sc)
        return np.array(peak_centered_scs), np.array(centralized_scs)
    
    def _get_sensor_pos(self,raw):
        # Get anterior-posterior sensor positions
        montage_positions = raw.get_montage().get_positions()['ch_pos']
        montage_positions = np.array([pos for pos in montage_positions.values()])
        r,a,s = montage_positions[:,0], montage_positions[:,1], montage_positions[:,2]
        return r,a,s