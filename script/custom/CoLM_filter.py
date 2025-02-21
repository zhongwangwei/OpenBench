import numpy as np
import pandas as pd
import re 
def adjust_time_CoLM(info, ds,syear,eyear,tim_res):
   match = re.match(r'(\d*)\s*([a-zA-Z]+)', tim_res)
   if match:
      # normalize time values
      num_value, time_unit = match.groups()
      num_value = int(num_value) if num_value else 1   
      ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-%dT%H:30:00'))
      if time_unit.lower() in ['m', 'month', 'mon']:
         # Handle river-related variables for monthly data
         if info.item in ['outflw', 'rivout', 'rivsto', 'rivout_inst', 'rivsto_inst', 
                         'rivdph', 'rivvel', 'fldout', 'fldsto', 'flddph', 'fldfrc',
                         'fldare', 'sfcelv', 'totout', 'totsto', 'storge', 'pthflw',
                         'pthout', 'gdwsto', 'gwsto', 'gwout', 'maxsto', 'maxflw',
                         'maxdph', 'damsto', 'daminf', 'wevap', 'winfilt', 'levsto', 'levdph']:
            if info.debug_mode:
               print('Adjusting time values for monthly river data...')
            # No time adjustment needed for monthly river data
            ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(days=15)
      elif time_unit.lower() in ['y', 'year', '1y', '1year']:
         pass
      elif time_unit.lower() in ['d', 'day', '1d', '1day']:
         if info.debug_mode:
            print('Adjusting time values for daily CoLM output...')
         ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(days=1)      
      elif time_unit.lower() in ['h', 'hour', '1h', '1hour']:
         if info.debug_mode:
            print('Adjusting time values for yearly CoLM output ...')
         ds['time'] = pd.DatetimeIndex(ds['time'].values) - pd.DateOffset(hours=1)
   else:
      print('tim_res error')
      exit()
   return ds

def filter_CoLM(info,ds):   #update info as well
   if info.item == "Crop_Yield_Corn":
      try:
         ds['Crop_Yield_Corn'] = (
            ((ds['f_cropprodc_rainfed_temp_corn'].fillna(0) * ds['area_rainfed_temp_corn'].fillna(0)) +
             (ds['f_cropprodc_irrigated_temp_corn'].fillna(0) * ds['area_irrigated_temp_corn'].fillna(0)) +
             (ds['f_cropprodc_rainfed_trop_corn'].fillna(0) * ds['area_rainfed_trop_corn'].fillna(0)) +
             (ds['f_cropprodc_irrigated_trop_corn'].fillna(0) * ds['area_irrigated_trop_corn'].fillna(0))) *
            (10**6) * 2.5 * (10**(-6)) /
            (ds['area_rainfed_temp_corn'].fillna(0) + ds['area_irrigated_temp_corn'].fillna(0) +
             ds['area_rainfed_trop_corn'].fillna(0) + ds['area_irrigated_trop_corn'].fillna(0)) *
            (3600. * 24. * 365.)) / 100.
      except:
         print("Missing variables:")
         missing_vars = []
         required_vars = [
              'f_cropprodc_rainfed_temp_corn', 'area_rainfed_temp_corn',
              'f_cropprodc_irrigated_temp_corn', 'area_irrigated_temp_corn',
              'f_cropprodc_rainfed_trop_corn', 'area_rainfed_trop_corn',
              'f_cropprodc_irrigated_trop_corn', 'area_irrigated_trop_corn'
          ]
         for var in required_vars:
            if var not in ds.variables:
               missing_vars.append(var)
         if missing_vars:
            print(f"Missing variables: {', '.join(missing_vars)}")
            return info, None
          #add the unit t ha-1 to dsout
      ds['Crop_Yield_Corn'] =  ds['Crop_Yield_Corn'].assign_attrs(varunit='t ha-1')
      info.sim_varname=['Crop_Yield_Corn']
      info.sim_varunit='t ha-1'
      return info, ds['Crop_Yield_Corn']

   if info.item == "Crop_Yield_Maize":
      try:
         ds['Crop_Yield_Maize']=(((ds['f_cropprodc_rainfed_temp_corn'].fillna(0)*ds['area_rainfed_temp_corn'].fillna(0))+
                                    (ds['f_cropprodc_irrigated_temp_corn'].fillna(0)*ds['area_irrigated_temp_corn'].fillna(0)) +
                                    (ds['f_cropprodc_rainfed_trop_corn'].fillna(0)*ds['area_rainfed_trop_corn'].fillna(0)) +
                                    (ds['f_cropprodc_irrigated_trop_corn'].fillna(0)*ds['area_irrigated_trop_corn'].fillna(0)))*
                                    (10**6)*2.5*(10**(-6))/(ds['area_rainfed_temp_corn'].fillna(0)+ds['area_irrigated_temp_corn'].fillna(0)+ds['area_rainfed_trop_corn'].fillna(0)+
                                    ds['area_irrigated_trop_corn'].fillna(0))*(3600.*24.*365.))/100.
      except:
         print("Missing variables:")
         missing_vars = []
         required_vars = [
              'f_cropprodc_rainfed_temp_corn', 'area_rainfed_temp_corn',
              'f_cropprodc_irrigated_temp_corn', 'area_irrigated_temp_corn',
              'f_cropprodc_rainfed_trop_corn', 'area_rainfed_trop_corn',
              'f_cropprodc_irrigated_trop_corn', 'area_irrigated_trop_corn'
          ]
         for var in required_vars:
            if var not in ds.variables:
               missing_vars.append(var)
         if missing_vars:
            print(f"Missing variables: {', '.join(missing_vars)}")
            return info, None
          #add the unit t ha-1 to dsout
      ds['Crop_Yield_Maize'] =  ds['Crop_Yield_Maize'].assign_attrs(varunit='t ha-1')

      info.sim_varname=['Crop_Yield_Maize']
      info.sim_varunit='t ha-1'
      return info, ds['Crop_Yield_Maize']

   if info.item == "Crop_Yield_Soybean":
      try:
         ds['Crop_Yield_Soybean']=(((ds['f_cropprodc_rainfed_temp_soybean'].fillna(0)*ds['area_rainfed_temp_soybean'].fillna(0))+
                                    (ds['f_cropprodc_irrigated_temp_soybean'].fillna(0)*ds['area_irrigated_temp_soybean'].fillna(0)) +
                                    (ds['f_cropprodc_rainfed_trop_soybean'].fillna(0)*ds['area_rainfed_trop_soybean'].fillna(0)) +
                                    (ds['f_cropprodc_irrigated_trop_soybean'].fillna(0)*ds['area_irrigated_trop_soybean'].fillna(0)))*
                                    (10**6)*2.5*(10**(-6))/(ds['area_rainfed_temp_soybean'].fillna(0)+ds['area_irrigated_temp_soybean'].fillna(0)+ds['area_rainfed_trop_soybean'].fillna(0)+
                                    ds['area_irrigated_trop_soybean'].fillna(0))*(3600.*24.*365.))/100.
      except:
         print("Missing variables:")
         missing_vars = []
         required_vars = [
              'f_cropprodc_rainfed_temp_soybean', 'area_rainfed_temp_soybean',
              'f_cropprodc_irrigated_temp_soybean', 'area_irrigated_temp_soybean',
              'f_cropprodc_rainfed_trop_soybean', 'area_rainfed_trop_soybean',
              'f_cropprodc_irrigated_trop_soybean', 'area_irrigated_trop_soybean'
          ]
         for var in required_vars:
            if var not in ds.variables:
               missing_vars.append(var)
         if missing_vars:
            print(f"Missing variables: {', '.join(missing_vars)}")
            return info, None
          #add the unit t ha-1 to dsout
      ds['Crop_Yield_Soybean'] =  ds['Crop_Yield_Soybean'].assign_attrs(varunit='t ha-1')

      info.sim_varname=['Crop_Yield_Soybean']
      info.sim_varunit='t ha-1'
      return info, ds['Crop_Yield_Soybean']

   if info.item == "Crop_Yield_Rice":
      try:
         ds['Crop_Yield_Rice']=(((ds['f_cropprodc_rainfed_rice']*ds['area_rainfed_rice'])+
                                    (ds['f_cropprodc_irrigated_rice']*ds['area_irrigated_rice']))*(10**6)*2.5*(10**(-6))/(ds['area_rainfed_rice']+ds['area_irrigated_rice'])*(3600.*24.*365.))/100.
      except:
         print("Missing variables:")
         missing_vars = []
         required_vars = [
              'f_cropprodc_rainfed_rice', 'area_rainfed_rice',
              'f_cropprodc_irrigated_rice', 'area_irrigated_rice'
          ]
         for var in required_vars:
            if var not in ds.variables:
               missing_vars.append(var)
         if missing_vars:
            print(f"Missing variables: {', '.join(missing_vars)}")
            return info, None
          #add the unit t ha-1 to dsout
      ds['Crop_Yield_Rice'] =  ds['Crop_Yield_Rice'].assign_attrs(varunit='t ha-1')

      info.sim_varname=['Crop_Yield_Rice']
      info.sim_varunit='t ha-1'
      return info, ds['Crop_Yield_Rice']

   if info.item == "Crop_Yield_Wheat":
      try:
         ds['Crop_Yield_Wheat']=(((ds['f_cropprodc_rainfed_spwheat'].fillna(0)*ds['area_rainfed_spwheat'].fillna(0))+
                                    (ds['f_cropprodc_irrigated_spwheat'].fillna(0)*ds['area_irrigated_spwheat'].fillna(0))+
                                    (ds['f_cropprodc_rainfed_wtwheat'].fillna(0)*ds['area_rainfed_wtwheat'].fillna(0))+
                                    (ds['f_cropprodc_irrigated_wtwheat'].fillna(0)*ds['area_irrigated_wtwheat'].fillna(0)))*
                                    (10**6)*2.5*(10**(-6))/(ds['area_rainfed_spwheat'].fillna(0)+ds['area_irrigated_spwheat'].fillna(0)+
                                    ds['area_rainfed_wtwheat'].fillna(0)+ds['area_irrigated_wtwheat'].fillna(0))*(3600.*24.*365.))/100.
      except:
         print("Missing variables:")
         missing_vars = []
         required_vars = [
              'f_cropprodc_rainfed_spwheat', 'area_rainfed_spwheat',
              'f_cropprodc_irrigated_spwheat', 'area_irrigated_spwheat',
              'f_cropprodc_rainfed_wtwheat','area_rainfed_wtwheat',
              'f_cropprodc_irrigated_wtwheat','area_irrigated_wtwheat'
          ]
         for var in required_vars:
            if var not in ds.variables:
               missing_vars.append(var)
         if missing_vars:
            print(f"Missing variables: {', '.join(missing_vars)}")
            return info, None
          #add the unit t ha-1 to dsout
      ds['Crop_Yield_Wheat'] =  ds['Crop_Yield_Wheat'].assign_attrs(varunit='t ha-1')

      info.sim_varname=['Crop_Yield_Wheat']
      info.sim_varunit='t ha-1'
      return info, ds['Crop_Yield_Wheat']

   if info.item == "Canopy_Interception":
      try:
         ds['Canopy_Interception']=ds['f_fevpl']-ds['f_etr']
         info.sim_varname=['Canopy_Interception']
         info.sim_varunit=' mm s-1'
      except:
         print('canopy interception evaporation calculation processing ERROR!!!')
      return info, ds['Canopy_Interception']

   if info.item == "Precipitation":
      try:
         if 'Precipitation' in ds.variables:
               # Use method='nearest' to select the nearest value in the 'soil' index
               ds['Precipitation'] = ds['Precipitation']
               info.sim_varname = ['Precipitation']
               info.sim_varunit = 'mm s-1'
         else:
               ds['Precipitation']=ds['f_xy_rain']+ds['f_xy_snow']
      except Exception as e:
         print(f"Surface Precipitation calculation processing ERROR: {e}")
         return info, None
      return info, ds['Precipitation']
      

   if info.item == "Surface_Net_SW_Radiation":
      try:
         ds['Surface_Net_SW_Radiation']=ds['f_xy_solarin']- ds['f_sr']
         info.sim_varname=['Surface_Net_SW_Radiation']
         info.sim_varunit='W m-2'
      except:
         print('Surface Net SW Radiation calculation processing ERROR!!!')
      return info, ds['Surface_Net_SW_Radiation']
   
   if info.item == "Surface_Net_LW_Radiation":
      try:
         ds['Surface_Net_LW_Radiation']=ds['f_xy_frl']-ds['f_olrg']
         info.sim_varname=['Surface_Net_LW_Radiation']
         info.sim_varunit='W m-2'
      except:
         print('Surface Net LW Radiation calculation processing ERROR!!!')
      return info, ds['Surface_Net_LW_Radiation']
   
   if info.item == "Surface_Soil_Moisture":
      try:
            # Use method='nearest' to select the nearest value in the 'soil' index
            try:
               ds['f_wliq_soisno']= (ds['f_wliq_soisno'].isel(soilsnow=5) +
                                     ds['f_wliq_soisno'].isel(soilsnow=6))/0.0626/1000.0
            except:
               ds['f_wliq_soisno']= (ds['f_wliq_soisno'].isel(soil_snow_lev=5) +
                                     ds['f_wliq_soisno'].isel(soil_snow_lev=6))/0.0626/1000.0
      
            info.sim_varname = ['f_wliq_soisno']
            info.sim_varunit = 'unitless'

      except Exception as e:
         print(f"Surface soil moisture calculation processing ERROR: {e}")
         return info, None
      return info, ds['f_wliq_soisno']
   

   if info.item == "Root_Zone_Soil_Moisture":
      try:
            try:
               ds['f_wliq_soisno']= (ds['f_wliq_soisno'].isel(soilsnow=5) +
                                    ds['f_wliq_soisno'].isel(soilsnow=6)+
                                    ds['f_wliq_soisno'].isel(soilsnow=7)+
                                    ds['f_wliq_soisno'].isel(soilsnow=8)+
                                    ds['f_wliq_soisno'].isel(soilsnow=9)+
                                    ds['f_wliq_soisno'].isel(soilsnow=10)+
                                    ds['f_wliq_soisno'].isel(soilsnow=11)+
                                    ds['f_wliq_soisno'].isel(soilsnow=12)*0.31
                                          )/1000.0
            except:
               ds['f_wliq_soisno']= (ds['f_wliq_soisno'].isel(soil_snow_lev=5) +
                                    ds['f_wliq_soisno'].isel(soil_snow_lev=6)+
                                    ds['f_wliq_soisno'].isel(soil_snow_lev=7)+
                                    ds['f_wliq_soisno'].isel(soil_snow_lev=8)+
                                    ds['f_wliq_soisno'].isel(soil_snow_lev=9)+
                                    ds['f_wliq_soisno'].isel(soil_snow_lev=10)+
                                    ds['f_wliq_soisno'].isel(soil_snow_lev=11)+
                                    ds['f_wliq_soisno'].isel(soil_snow_lev=12)*0.31
                                          )/1000.0               
            info.sim_varname = ['f_wliq_soisno']
            info.sim_varunit = 'unitless'
      except Exception as e:
         print(f"Surface soil moisture calculation processing ERROR: {e}")
         return info, None
      return info, ds['f_wliq_soisno']
   if info.item == "Albedo":
      try:
            ds['Albedo']= ds['f_sr'] /  ds['f_xy_solarin'] 

            info.sim_varname = ['Albedo']
            info.sim_varunit = 'unitless'
      except Exception as e:
         print(f"Surface Albedo calculation processing ERROR: {e}")
         return info, None
      return info, ds['Albedo']
   if info.item == "Terrestrial_Water_Storage_Change":
      #f_xxx is the average value, and several variables with ***_inst are instantaneous
      #need to double check the calculation method
      try:
         w1=ds['f_wat'][1:,:,:]+ds['f_wa'][1:,:,:]+ds['f_wdsrf'][1:,:,:]+ds['f_wetwat'][1:,:,:]
         w0=ds['f_wat'][:-1,:,:]+ds['f_wa'][:-1,:,:]+ds['f_wdsrf'][:-1,:,:]+ds['f_wetwat'][:-1,:,:]
         tmp=w1-w0
         #set first time step of Terrestrial_Water_Storage_Change to nan, and then set other time steps to tmp
         ds['Terrestrial_Water_Storage_Change']=ds['f_wat'].copy()
         ds['Terrestrial_Water_Storage_Change'][0,:,:]=np.nan
         ds['Terrestrial_Water_Storage_Change'][1:,:,:]=tmp
         info.sim_varname=['Terrestrial_Water_Storage_Change']
         info.sim_varunit='mm'
      except:
         print('Terrestrial Water Storage Change calculation processing ERROR!!!')
         return info, None
      return info, ds['Terrestrial_Water_Storage_Change']


   if info.item == "Surface_Wind_Speed":
      try:
            ds['Surface_Wind_Speed']= (ds['f_us10m']**2+ds['f_vs10m']**2)**0.5
            info.sim_varname = ['Surface_Wind_Speed']
            info.sim_varunit = 'm s-1 wind'
      except Exception as e:
         print(f"Surface Wind Speed calculation processing ERROR: {e}")
         return info, None
      return info, ds['Surface_Wind_Speed']


   if info.item == "Urban_Anthropogenic_Heat_Flux":
      try:
         ds['Urban_Anthropogenic_Heat_Flux']=ds['f_fhac']+ds['f_fach']+ds['f_fhah']+ds['f_fvehc']+ds['f_fmeta']+ds['f_fwst']
         info.sim_varname=['Urban_Anthropogenic_Heat_Flux']
         info.sim_varunit='W m-2'
      except Exception as e:
         print(f"Urban Anthropogenic Heat Flux calculation processing ERROR: {e}")
         return info, None
      return info, ds['Urban_Anthropogenic_Heat_Flux']

   if info.item == "Terrestrial_Water_Storage_Change":
      try:
         TWS=ds['f_wat']+ds['f_wa']+ds['f_wdsrf']+ds['f_wetwat']
         # Initialize TWSC with same shape as TWS but filled with NaN values
         ds['Terrestrial_Water_Storage_Change'] = TWS.copy()
         ds['Terrestrial_Water_Storage_Change'][0,:,:] = np.nan
         # Calculate TWSC as difference between consecutive time steps
         ds['Terrestrial_Water_Storage_Change'][1:,:,:] = TWS[1:,:,:] - TWS[:-1,:,:]
         info.sim_varname=['Terrestrial_Water_Storage_Change']
         info.sim_varunit='mm'
      except:
         print('Terrestrial Water Storage Change calculation processing ERROR!!!')
