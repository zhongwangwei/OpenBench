# -*- coding: utf-8 -*-
import numpy as np
import sys
import xarray as xr
import logging
class metrics:
    """
    A class for calculating various statistical metrics for model evaluation.
    """

    def __init__(self):
        """
        Initialize the Metrics class with metadata.
        """
        self.name = 'metrics'
        self.version = '0.2'
        self.release = '0.2'
        self.date = 'March 2024'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        
        # Suppress numpy warnings
        np.seterr(all='ignore')

    def _validate_inputs(self, s, o):
        """
        Validate and align input DataArrays.

        Args:
            s (xr.DataArray): Simulated data
            o (xr.DataArray): Observed data

        Returns:
            tuple: Aligned and validated DataArrays
        """
        # Ensure inputs are xarray DataArrays
        if not isinstance(s, xr.DataArray) or not isinstance(o, xr.DataArray):
            logging.error('Inputs must be xarray DataArrays')
            raise TypeError("Inputs must be xarray DataArrays")

        # Align time dimensions
        s, o = xr.align(s, o, join='inner')

        # Remove NaN values
        mask = np.isfinite(s) & np.isfinite(o)
        return s.where(mask), o.where(mask)
    
    def percent_bias(self, s, o):
        """
        Calculate Percent Bias.

        Args:
            s (xr.DataArray): Simulated data
            o (xr.DataArray): Observed data

        Returns:
            xr.DataArray: Percent bias
        """
        s, o = self._validate_inputs(s, o)
        return 100.0 * (s - o).sum(dim='time') / o.sum(dim='time')


    def absolute_percent_bias(self, s, o):
        """
        Calculate Absolute Percent Bias (APB).

        Args:
            s (xr.DataArray): Simulated data
            o (xr.DataArray): Observed data

        Returns:
            xr.DataArray: Absolute percent bias
        """
        # Validate and align inputs
        s, o = self._validate_inputs(s, o)
        
        # Calculate absolute percent bias
        apb = 100.0 * abs((s - o).sum(dim='time')) / o.sum(dim='time')
        return apb

    def RMSE(self, s, o):
        """
        Calculate Root Mean Squared Error (RMSE).

        Args:
            s (xr.DataArray): Simulated data
            o (xr.DataArray): Observed data

        Returns:
            xr.DataArray: Root mean squared error
        """
        # Validate and align inputs
        s, o = self._validate_inputs(s, o)
        
        # Calculate RMSE
        rmse = np.sqrt(((s - o) ** 2).mean(dim='time'))
        return rmse
    
    def ubRMSE(self, s, o):
        """
        Calculate Unbiased Root Mean Squared Error (ubRMSE).

        Args:
            s (xr.DataArray): Simulated data
            o (xr.DataArray): Observed data

        Returns:
            xr.DataArray: Unbiased root mean squared error
        """
        # Validate and align inputs
        s, o = self._validate_inputs(s, o)
        
        # Calculate unbiased RMSE
        ubrmse = np.sqrt((((s - s.mean(dim='time')) - (o - o.mean(dim='time'))) ** 2).mean(dim='time'))
        return ubrmse
 
    def CRMSD(self, s, o=None):
        """
        Calculate Centered Root Mean Square Difference (CRMSD).

        Args:
            s (xr.DataArray): Simulated data
            o (xr.DataArray, optional): Observed data. If not provided, the mean along the time dimension is used as the reference.

        Returns:
            xr.DataArray: Centered root mean square difference
        """
        # If observed data is not provided, use the mean of simulated data as reference
        if o is None:
            o = s.mean(dim="time")
        
        # Validate and align inputs
        s, o = self._validate_inputs(s, o)
        
        # Calculate standard deviations
        std_s = s.std(dim="time")
        std_o = o.std(dim="time")

        # Calculate correlations
        correlations = xr.corr(s, o, dim="time")

        # Apply the CRMSD formula
        crmsd = np.sqrt(std_s**2 + std_o**2 - 2 * std_s * std_o * correlations)
        return crmsd
    
    def mean_absolute_error(self,s,o):
        """
        Mean Absolute Error
        input:
            s: simulated
            o: observed
        output:
            maes: mean absolute error
        """
        #np.mean(abs(self.s-self.o))
        k1=s-o
        var=(abs(k1)).mean(dim='time')
        return var

    def bias(self,s,o):
        """
        Bias
        input:
            s: simulated
            o: observed
        output:
            bias: bias
        """
        #np.mean(s-o)
        var=(s-o).mean(dim='time')
        return var

    def L(self,s,o,N=5):
        """
        Likelihood
        input:
            s: simulated
            o: observed
        output:
            L: likelihood
        """
        #np.exp(-N*sum((self.s-self.o)**2)/sum((self.o-np.mean(self.o))**2))
        tmp1=((o-o.mean(dim='time'))**2).sum(dim='time')
        tmp2=-N*(((s-o)**2).sum(dim='time'))
        var=np.exp(tmp2/tmp1)
        return var 

    def correlation(self,s,o):
        """
        correlation coefficient
        input:
            s: simulated
            o: observed
        output:
            correlation: correlation coefficient
        """
        corr=xr.corr(s,o,dim=['time'])
      
        return corr

    def correlation_R2(self,s,o):
        """
        correlation coefficient R2
        input:
            s: simulated
            o: observed
        output:
            correlation: correlation coefficient
        """

        return xr.corr(s,o,dim=['time'])**2
    
    def NSE(self,s,o):
        """
        Nash Sutcliffe efficiency coefficient
        input:
            s: simulated
            o: observed
        output:
            nse: Nash Sutcliffe efficient coefficient
        """
        #1 - sum((s-o)**2)/sum((o-np.mean(o))**2)
        _tmp1=((o-o.mean(dim='time'))**2).sum(dim='time')
        _tmp2=((s-o)**2).sum(dim='time')
        var=1-_tmp2/_tmp1
        return var
    
    def KGE(self,s,o):
        """
        Kling-Gupta Efficiency
        input:
            s: simulated
            o: observed
        output:
            kge: Kling-Gupta Efficiency
            cc: correlation
            alpha: ratio of the standard deviation
            beta: ratio of the mean
        """
        cc  = self.correlation(s,o)
        alpha =s.std(dim='time')/o.std(dim='time')
        #alpha = np.std(s)/np.std(o)
        beta = s.mean(dim='time')/o.mean(dim='time')
        #beta = np.sum(s)/np.sum(o)
        kge = 1- ( (cc-1)**2 + (alpha-1)**2 + (beta-1)**2 )**0.5
        return kge   #, cc, alpha, beta

    def KGESS(self,s,o):
        """
        Normalized Kling-Gupta Efficiency
        input:
            s: simulated
            o: observed
        output:
            kgess:Normalized Kling-Gupta Efficiency
        note:
        KGEbench= −0.41 from Knoben et al., 2019)
        Knoben, W. J. M., Freer, J. E., and Woods, R. A.: Technical note: Inherent benchmark or not? Comparing Nash–Sutcliffe and Kling–
        Gupta efficiency scores, Hydrol. Earth Syst. Sci., 23, 4323–4331,
        https://doi.org/10.5194/hess-23-4323-2019, 2019.
        """
        kge=self.KGE(s,o)
        kgess=(kge-(-0.41))/(1.0-(-0.41))
        return kgess   #, cc, alpha, beta

    def index_agreement(self,s,o):
        """
	    index of agreement
	    input:
            s: simulated
            o: observed
        output:
            ia: index of agreement
        """
        _tmp1 = ((o-s)**2).sum(dim='time')
        _tmp2 =((np.abs(s-o.mean(dim='time'))+np.abs(o-o.mean(dim='time')))**2).sum(dim='time')
        ia = 1 - _tmp1/_tmp2
        return ia.squeeze()

    def kappa_coeff(self,s,o):
        s = (s).astype(int)
        o = (o).astype(int)
        n = len(s)
        foo1 = np.unique(s)
        foo2 = np.unique(o)
        unique_data = np.unique(np.hstack([foo1,foo2]).flatten())
        self.unique_data = unique_data
        kappa_mat = np.zeros((len(unique_data),len(unique_data)))
        ind1 = np.empty(n, dtype=int)
        ind2 = np.empty(n, dtype=int)
        for i in range(len(unique_data)):
            ind1[s==unique_data[i]] = i
            ind2[o==unique_data[i]] = i
        for i in range(n):
            kappa_mat[ind1[i],ind2[i]] += 1
        self.kappa_mat = kappa_mat
        # compute kappa coefficient
        # formula for kappa coefficient taken from
        # http://adorio-research.org/wordpress/?p=2301
        tot = np.sum(kappa_mat)
        Pa = np.sum(np.diag(kappa_mat))/tot
        PA = np.sum(kappa_mat,axis=0)/tot
        PB = np.sum(kappa_mat,axis=1)/tot
        Pe = np.sum(PA*PB)
        kappa_coeff = (Pa-Pe)/(1-Pe)

        return kappa_mat, kappa_coeff

    def rv(self,s,o):
        '''
        Relative variability
        (or amplitude ratio)
        input:
            s: simulated
            o: observed
        output:
            rv : relative variability, amplitude ratio 
        Reference:
        ****
        '''
        return s.std(dim='time') / o.std(dim='time') - 1.0

    def ubNSE(self,s,o):
        """
        Unbiased Nash Sutcliffe efficiency coefficient
        input:
            s: simulated
            o: observed
        output:
            ubnse: Unbiased Nash Sutcliffe efficient coefficient
        """
        _tmp1=((o-o.mean(dim='time'))**2).sum(dim='time')
        _tmp2=(((s - s.mean(dim='time')) - (o - o.mean(dim='time')))**2).sum(dim='time')
        var=1-_tmp2/_tmp1
        return var

    def ubKGE(self,s,o):
        """
        Unbiased Kling-Gupta Efficiency
        input:
            s: simulated
            o: observed
        output:
            kge: Kling-Gupta Efficiency
   
        """
        s,o = self.rm_mean(s,o)
        var = self.KGE(s,o)
        return var

    def ubcorrelation(self,s,o):
        """
        correlation coefficient
        input:
            s: simulated
            o: observed
        output:
            correlation: correlation coefficient
        """
        s,o = self.rm_mean(s,o)
        var = self.correlation(s,o)
        return var

    def ubcorrelation_R2(self,s,o):
        """
        correlation coefficient R2
        input:
            s: simulated
            o: observed
        output:
            correlation: correlation coefficient
        """
        s,o = self.rm_mean(s,o)
        var = self.correlation_R2(s,o)
        return var
    
    def rm_mean(self,s,o):
        t1 = s - min(s.min(dim='time'),o.min(dim='time'))
        t2 = o - min(s.min(dim='time'),o.min(dim='time'))
        s = t1
        o = t2
        return s,o

    def pc_max(self,s,o):
        mask1 = np.isnan(s) | np.isnan(o)
        s.values[mask1] = np.nan
        o.values[mask1] = np.nan
        #remove the nan values
        s=s.dropna(dim='time').astype(np.float32)
        o=o.dropna(dim='time').astype(np.float32)

        return  (s.max(dim='time')-o.max(dim='time'))/o.max(dim='time')      #(np.max(s)-np.max(o))/np.max(o)

    def pc_min(self,s,o):

        mask1 = np.isnan(s) | np.isnan(o)
        s.values[mask1] = np.nan
        o.values[mask1] = np.nan
        #remove the nan values
        s=s.dropna(dim='time').astype(np.float32)
        o=o.dropna(dim='time').astype(np.float32)

        return (s.min(dim='time')-o.min(dim='time'))/o.min(dim='time')    #(np.min(s)-np.min(o))/np.min(o)

    def pc_ampli(self,s,o):
        mask1 = np.isnan(s) | np.isnan(o)
        s.values[mask1] = np.nan
        o.values[mask1] = np.nan
        #remove the nan values
        s=s.dropna(dim='time').astype(np.float32)
        o=o.dropna(dim='time').astype(np.float32)

        return (s.max(dim='time')-s.min(dim='time'))/ (o.max(dim='time')- o.min(dim='time')) -1.0 #(np.max(s) - np.min(s)) / (np.max(o) - np.min(o)) - 1.0
   
    def rSD(self,s,o):
        #Ratio of standard deviations
        #also see E. Towler et al.: Benchmarking model simulations of retrospective streamflow in the contiguous US
        #Indicates if flow variability is being over- or underestimated; calculated from rSD in the hydroGOF R package
        pass
    
    def PBIAS_HF (self,s,o):
        #Percent bias of flows ≥ Q98 (Yilmaz et al., 2008)
        #also see E. Towler et al.: Benchmarking model simulations of retrospective streamflow in the contiguous US
        #Characterizes response to large precipitation events; calculated using flows ≥ the 98th percentile flow with pbias in the hydroGOF R package
        pass
    
    def PBIAS_LF (self,s,o):
        #Percent bias of flows ≤ Q30(Yilmaz et al., 2008)
        #also see E. Towler et al.: Benchmarking model simulations of retrospective streamflow in the contiguous US
        #Characterizes baseflow; calculated following equations in
        #Yilmaz et al. (2008) using logged flows ≤ the 30th percentile (zeros are set to the USGS observational threshold
        #of 0.01 ft3 s−1 (0.000283 m3 s−1))
        pass

    def APFB(self, data_array, obs_array, start_month=1, out_per_year=False, fun=None, epsilon_type="none", epsilon_value=None):
        """
        Calculates the Annual Peak Flow Bias (APFB) along the time dimension of Xarray DataArrays.

        Args:
            data_array (xr.DataArray): Simulated data.
            obs_array (xr.DataArray): Observed data.
            start_month (int, optional): Starting month of the hydrological year (1-12). Defaults to 1 (January).
            out_per_year (bool, optional): If True, returns APFB per year. Defaults to False.
            fun (function, optional): Transformation function to apply to data before calculation. Defaults to None.
            epsilon_type (str, optional): Type of epsilon handling for zero values in 'fun'. Defaults to "none".
            epsilon_value (float, optional): Value for epsilon handling. Defaults to None.

        Returns:
            float or dict: Mean APFB or a dictionary with mean APFB and yearly APFB values.
        """

        # Align and handle missing values
        common_time = obs_array.time.values.astype('datetime64[D]')
        data_array = data_array.sel(time=common_time)
        obs_array = obs_array.sel(time=common_time)
        valid_indices = np.isfinite(data_array) & np.isfinite(obs_array)
        data_array = data_array.where(valid_indices)
        obs_array = obs_array.where(valid_indices)

        # Convert to pandas for easier time grouping
        df_sim = data_array.to_pandas().to_frame(name='simulated')
        df_obs = obs_array.to_pandas().to_frame(name='observed')

        # Apply transformation function
        if fun is not None:
            if epsilon_type == "Pushpalatha2012":
                epsilon = df_obs["observed"].mean() / 100
            elif epsilon_type == "otherFactor":
                epsilon = df_obs["observed"].mean() * epsilon_value
            elif epsilon_type == "otherValue":
                epsilon = epsilon_value
            else:
                epsilon = 0

            df_sim["simulated"] = df_sim["simulated"].apply(lambda x: fun(x + epsilon))
            df_obs["observed"] = df_obs["observed"].apply(lambda x: fun(x + epsilon))

        # Group by hydrological year and calculate peak flows
        df_sim['year'] = df_sim.index.to_period(f'{start_month}MS').year
        df_obs['year'] = df_obs.index.to_period(f'{start_month}MS').year
        annual_peaks_sim = df_sim.groupby('year')['simulated'].max()
        annual_peaks_obs = df_obs.groupby('year')['observed'].max()

        # Calculate APFB for each year
        apfb_per_year = (annual_peaks_sim - annual_peaks_obs) / annual_peaks_obs

        if out_per_year:
            return {"APFB_value": apfb_per_year.mean(), "APFB_per_year": apfb_per_year}
        else:
            return apfb_per_year.mean()
        
    def br2(self, data_array, obs_array, na_rm=True, use_abs=False, fun=None, epsilon_type="none", epsilon_value=None):
        """
        Calculates the br2 metric (R-squared multiplied by regression slope) along the time dimension.

        Args:
            data_array (xr.DataArray): Simulated data.
            obs_array (xr.DataArray): Observed data.
            na_rm (bool, optional): If True, removes missing values before calculation. Defaults to True.
            use_abs (bool, optional): If True, uses absolute value of slope in calculation. Defaults to False.
            fun (function, optional): Transformation function to apply to data before calculation. Defaults to None.
            epsilon_type (str, optional): Type of epsilon handling for zero values in 'fun'. Defaults to "none".
            epsilon_value (float, optional): Value for epsilon handling. Defaults to None.

        Returns:
            xr.DataArray: An array containing the br2 values for each time step.
        """

        # Align and handle missing values
        common_time = obs_array.time.values.astype('datetime64[D]')
        data_array = data_array.sel(time=common_time)
        obs_array = obs_array.sel(time=common_time)
        valid_indices = np.isfinite(data_array) & np.isfinite(obs_array)
        data_array = data_array.where(valid_indices)
        obs_array = obs_array.where(valid_indices)

        # Apply transformation function
        if fun is not None:
            if epsilon_type == "Pushpalatha2012":
                epsilon = obs_array.mean(dim="time") / 100
            elif epsilon_type == "otherFactor":
                epsilon = obs_array.mean(dim="time") * epsilon_value
            elif epsilon_type == "otherValue":
                epsilon = epsilon_value
            else:
                epsilon = 0

            data_array = fun(data_array + epsilon)
            obs_array = fun(obs_array + epsilon)

        # Calculate R-squared and regression slope
        def calculate_for_single_time(sim, obs):
            r_squared = np.corrcoef(sim, obs)[0, 1]**2
            slope, _, _, _, _ = linregress(obs, sim)  # Force intercept to zero
            if use_abs:
                slope = abs(slope)
            br2_value = r_squared * slope if slope <= 1 else r_squared / slope
            return br2_value

        br2_values = xr.apply_ufunc(
            calculate_for_single_time, data_array, obs_array,
            input_core_dims=[["time"], ["time"]],
            vectorize=True,
            dask="parallelized"
        )

        return br2_values

    def cp(self, data_array, obs_array, fun=None, epsilon_type="none", epsilon_value=None):
        """
        Calculates the Coefficient of Persistence (CP) along the time dimension of Xarray DataArrays.

        Args:
            data_array (xr.DataArray): Simulated data.
            obs_array (xr.DataArray): Observed data.
            fun (function, optional): Transformation function to apply to data before calculation. Defaults to None.
            epsilon_type (str, optional): Type of epsilon handling for zero values in 'fun'. Defaults to "none".
            epsilon_value (float, optional): Value for epsilon handling. Defaults to None.

        Returns:
            xr.DataArray: An array containing the CP values for each time step.
        """

        # Align and handle missing values
        common_time = obs_array.time.values.astype('datetime64[D]')
        data_array = data_array.sel(time=common_time)
        obs_array = obs_array.sel(time=common_time)
        valid_indices = np.isfinite(data_array) & np.isfinite(obs_array)
        data_array = data_array.where(valid_indices)
        obs_array = obs_array.where(valid_indices)

        # Apply transformation function
        if fun is not None:
            if epsilon_type == "Pushpalatha2012":
                epsilon = obs_array.mean(dim="time") / 100
            elif epsilon_type == "otherFactor":
                epsilon = obs_array.mean(dim="time") * epsilon_value
            elif epsilon_type == "otherValue":
                epsilon = epsilon_value
            else:
                epsilon = 0

            data_array = fun(data_array + epsilon)
            obs_array = fun(obs_array + epsilon)

        # Calculate differences
        diff_sim_obs = data_array.diff(dim="time")
        diff_obs_obs = obs_array.diff(dim="time")

        # Calculate numerator and denominator
        numerator = (diff_sim_obs**2).sum(dim="time")
        denominator = (diff_obs_obs**2).sum(dim="time")

        # Calculate CP
        cp = 1 - (numerator / denominator)

        return cp

    def dr(self, data_array, obs_array, fun=None, epsilon_type="none", epsilon_value=None):
        """
        Calculates the Refined Index of Agreement (dr) along the time dimension of Xarray DataArrays.

        Args:
            data_array (xr.DataArray): Simulated data.
            obs_array (xr.DataArray): Observed data.
            fun (function, optional): Transformation function to apply to data before calculation. Defaults to None.
            epsilon_type (str, optional): Type of epsilon handling for zero values in 'fun'. Defaults to "none".
            epsilon_value (float, optional): Value for epsilon handling. Defaults to None.

        Returns:
            xr.DataArray: An array containing the dr values for each time step.
        """

        # Align and handle missing values
        common_time = obs_array.time.values.astype('datetime64[D]')
        data_array = data_array.sel(time=common_time)
        obs_array = obs_array.sel(time=common_time)
        valid_indices = np.isfinite(data_array) & np.isfinite(obs_array)
        data_array = data_array.where(valid_indices)
        obs_array = obs_array.where(valid_indices)

        # Apply transformation function
        if fun is not None:
            if epsilon_type == "Pushpalatha2012":
                epsilon = obs_array.mean(dim="time") / 100
            elif epsilon_type == "otherFactor":
                epsilon = obs_array.mean(dim="time") * epsilon_value
            elif epsilon_type == "otherValue":
                epsilon = epsilon_value
            else:
                epsilon = 0

            data_array = fun(data_array + epsilon)
            obs_array = fun(obs_array + epsilon)

        # Calculate differences and mean of observations
        diff = np.abs(data_array - obs_array)
        obs_mean = obs_array.mean(dim="time")

        # Calculate terms A and B
        A = diff.sum(dim="time")
        B = 2 * np.abs(obs_array - obs_mean).sum(dim="time")

        # Calculate dr
        dr = 1 - (A / B)
        dr = xr.where(A > B, 1 - (B / A), dr)  # Handle cases where A > B

        return dr

    def smpi(self,s,o, n_bootstrap=100):
        #Calculate the Single Model Performance Index (SMPI) 
        
        # Calculate observational variance
        obs_var = o.var(dim='time')
        # Calculate squared differences
        diff_squared = (s - o)**2
        # Normalize by observational variance
        normalized_diff = diff_squared / obs_var
        # Calculate SMPI without weighting
        smpi = normalized_diff.mean(dim=['time', 'lat', 'lon'])
        # Calculate SMPI with latitude weighting
        #note: We don't think the latitude weighting is necessary for the SMPI calculation
        #note: this will give too much weight to the poles
        #weights = np.cos(np.deg2rad(model.lat))
        #weights = weights / weights.sum()
        #weights = weights.expand_dims({'lon': mod.lon.size})
        #smpi = (normalized_diff* weights).sum()

       # Bootstrap for uncertainty estimation
        bootstrap_smpi = []
        normalized_diff_values = normalized_diff.values
        n_times = normalized_diff.sizes['time']
        for _ in range(n_bootstrap):
            bootstrap_indices = np.random.choice(n_times, size=n_times, replace=True)
            bootstrap_sample = normalized_diff_values[bootstrap_indices]
            bootstrap_smpi.append(np.mean(bootstrap_sample))
        
        bootstrap_smpi = np.array(bootstrap_smpi)
        smpi_lower, smpi_upper = np.percentile(bootstrap_smpi, [5, 95])
        
        return smpi, smpi_lower, smpi_upper
        
    




