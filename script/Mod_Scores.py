import numpy as np
import xarray as xr
logging.getLogger('xarray').setLevel(logging.WARNING)  # Suppress INFO messages from xarray
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress numpy runtime warnings
logging.getLogger('dask').setLevel(logging.WARNING)  # Suppress INFO messages from dask
class scores:
    """
    A class for calculating various performance scores for model evaluation.
    The score varies from 0~1, 1 being the best.
    
    """

    def __init__(self):
        self.name = 'scores'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        
        # Suppress all numpy warnings
        np.seterr(all='ignore')

    
    def _calculate_mean_and_anomalies(self, data):
        """
        Calculate mean and anomalies for a given dataset.
        
        Args:
            data (xarray.DataArray): Input data
        
        Returns:
            tuple: (mean, anomalies)
        """
        mean = data.mean(dim='time')
        anomalies = data.groupby('time.month') - data.groupby('time.month').mean('time')
        return mean, anomalies

    
    def index_agreement(self, s, o):
        """
        Calculate index of agreement.
        
        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data
        
        Returns:
            xarray.DataArray: Index of agreement
        """
        numerator = ((o - s) ** 2).sum(dim='time')
        denominator = ((np.abs(s - o.mean(dim='time')) + np.abs(o - o.mean(dim='time'))) ** 2).sum(dim='time')
        return 1 - numerator / denominator
    
    
    def nBiasScore(self, s, o):
        """
        Calculate normalized Bias Score.
        
        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data
        
        Returns:
            xarray.DataArray: Normalized Bias Score (0 to 1, 1 being best)
        """
        bias = s.mean(dim='time') - o.mean(dim='time')
        crms = np.sqrt(((o - o.mean(dim='time')) ** 2).mean(dim='time'))
        return np.exp(-np.abs(bias) / crms)
    
    
    def nRMSEScore(self, s, o):
        """
        Calculate normalized RMSE Score.
        
        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data
        
        Returns:
            xarray.DataArray: Normalized RMSE Score (0 to 1, 1 being best)
        """
        s_mean, o_mean = s.mean(dim='time'), o.mean(dim='time')
        crms = np.sqrt(((o - o_mean) ** 2).mean(dim='time'))
        crmse = np.sqrt((((s - s_mean) - (o - o_mean)) ** 2).mean(dim='time'))
        return np.exp(-crmse / crms)

    
    def nPhaseScore(self, s, o):
        """
        Calculate normalized Phase Score.
        
        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data
        
        Returns:
            xarray.DataArray: Normalized Phase Score (0 to 1, 1 being best)
        """
        ref_max_month = o.groupby('time.month').mean('time').idxmax('month')
        sim_max_month = s.groupby('time.month').mean('time').idxmax('month')
        phase_shift = (sim_max_month - ref_max_month) * 365 / 12
        return 0.5 * (1 + np.cos(2 * np.pi * phase_shift / 365))
    
    
    def nIavScore(self, s, o):
        """
        Calculate normalized Interannual Variability Score.
        
        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data
        
        Returns:
            xarray.DataArray: Normalized IAV Score (0 to 1, 1 being best)
        """
        _, s_anom = self._calculate_mean_and_anomalies(s)
        _, o_anom = self._calculate_mean_and_anomalies(o)
        
        s_iav = np.sqrt((s_anom ** 2).mean('time'))
        o_iav = np.sqrt((o_anom ** 2).mean('time'))
        
        return np.exp(-np.abs(s_iav - o_iav) / o_iav)
    
    
    def nSpatialScore(self,s,o):
        """
        Calculate normalized Spatial Score.
        
        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data
        
        Returns:
            xarray.DataArray: Normalized Spatial Score (0 to 1, 1 being best)
        """
        smean=s.mean(dim='time').squeeze()
        omean=o.mean(dim='time').squeeze()
     
        # Calculate the spatial correlation between reference and model
        #spatial_corr = np.corrcoef(smean.values.flatten(), omean.values.flatten())[0, 1]
        try:
            spatial_corr = xr.corr(smean,omean, dim=['lat', 'lon'])
        except:
            spatial_corr = xr.corr(smean,omean)

        # Calculate the spatial standard deviation for reference and model
        ref_std = omean.std().squeeze()
        sim_std = smean.std().squeeze()
        sigma=sim_std/ref_std
        # Calculate the spatial score
        spatial_score_0 = 2.0*(1+spatial_corr)/(sigma+1/sigma)**2
        #spatial_score   = xr.full_like(smean, spatial_score_0)
        spatial_score   = smean*0.0+spatial_score_0
        return spatial_score

    
    def Overall_Score(self,s,o):
        """
        Calculate Overall Score based on multiple metrics.
        
        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data
        
        Returns:
            xarray.DataArray: Overall Score
        """
        k=6
        bias_score    = self.nBiasScore(s, o)
        rmse_score    = self.nRMSEScore(s, o)
        phase_score   = self.nPhaseScore(s, o)   
        iav_score     = self.nIavScore(s, o)    
        #if iav_score is nan,then k=k-1
        if np.isnan(iav_score).all():
            k=k-1
            iav_score=0.0
        spatial_score = self.nSpatialScore(s, o) 

        if np.isnan(spatial_score).all():
            k=k-1
            spatial_score=0.0
        # Aggregate Scores (Adjust weights as per your ILAMB configuration)
        
        overall_score = (bias_score + 2 * rmse_score + phase_score + iav_score + spatial_score) / k
        return overall_score 

    
    def nSeasonalityScore(self, s, o):
        """
        Calculate normalized Seasonality Score.
        
        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data
        
        Returns:
            xarray.DataArray: Normalized Seasonality Score (0 to 1, 1 being best)
        """
        s_amp = s.groupby('time.month').max('time') - s.groupby('time.month').min('time')
        o_amp = o.groupby('time.month').max('time') - o.groupby('time.month').min('time')
        relative_error = (s_amp - o_amp) / o_amp
        return np.exp(-relative_error)
    


