def filter_CoLM(info,ds):   #update info as well
   if info.item == "Crop_Yield_Corn":
      ds = ds.fillna(0)
      try:
         ds['Crop_Yield_Corn']=(((ds['f_cropprodc_rainfed_temp_corn']*ds['area_rainfed_temp_corn'])+
                                    (ds['f_cropprodc_irrigated_temp_corn']*ds['area_irrigated_temp_corn']) +
                                    (ds['f_cropprodc_rainfed_trop_corn']*ds['area_rainfed_trop_corn']) +
                                    (ds['f_cropprodc_irrigated_trop_corn']*ds['area_irrigated_trop_corn']))*
                                    (10**6)*2.5*(10**(-6))/(ds['area_rainfed_temp_corn']+ds['area_irrigated_temp_corn']+ds['area_rainfed_trop_corn']+
                                    ds['area_irrigated_trop_corn'])*(3600.*24.*365.))/100.
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

      info.sim_varname='Crop_Yield_Corn'
      info.sim_varunit='t ha-1'
      return info, ds['Crop_Yield_Corn']

   if info.item == "Crop_Yield_Maize":
      ds = ds.fillna(0)
      try:
         ds['Crop_Yield_Maize']=(((ds['f_cropprodc_rainfed_temp_corn']*ds['area_rainfed_temp_corn'])+
                                    (ds['f_cropprodc_irrigated_temp_corn']*ds['area_irrigated_temp_corn']) +
                                    (ds['f_cropprodc_rainfed_trop_corn']*ds['area_rainfed_trop_corn']) +
                                    (ds['f_cropprodc_irrigated_trop_corn']*ds['area_irrigated_trop_corn']))*
                                    (10**6)*2.5*(10**(-6))/(ds['area_rainfed_temp_corn']+ds['area_irrigated_temp_corn']+ds['area_rainfed_trop_corn']+
                                    ds['area_irrigated_trop_corn'])*(3600.*24.*365.))/100.
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

      info.sim_varname='Crop_Yield_Maize'
      info.sim_varunit='t ha-1'
      return info, ds['Crop_Yield_Maize']

   if info.item == "Crop_Yield_Soybean":
      ds = ds.fillna(0)
      try:
         ds['Crop_Yield_Soybean']=(((ds['f_cropprodc_rainfed_temp_soybean']*ds['area_rainfed_temp_soybean'])+
                                    (ds['f_cropprodc_irrigated_temp_soybean']*ds['area_irrigated_temp_soybean']) +
                                    (ds['f_cropprodc_rainfed_trop_soybean']*ds['area_rainfed_trop_soybean']) +
                                    (ds['f_cropprodc_irrigated_trop_soybean']*ds['area_irrigated_trop_soybean']))*
                                    (10**6)*2.5*(10**(-6))/(ds['area_rainfed_temp_soybean']+ds['area_irrigated_temp_soybean']+ds['area_rainfed_trop_soybean']+
                                    ds['area_irrigated_trop_soybean'])*(3600.*24.*365.))/100.
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

      info.sim_varname='Crop_Yield_Soybean'
      info.sim_varunit='t ha-1'
      return info, ds['Crop_Yield_Soybean']

   if info.item == "Crop_Yield_Rice":
      ds = ds.fillna(0)
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

      info.sim_varname='Crop_Yield_Rice'
      info.sim_varunit='t ha-1'
      return info, ds['Crop_Yield_Rice']

   if info.item == "Crop_Yield_Wheat":
      ds = ds.fillna(0)
      try:
         ds['Crop_Yield_Wheat']=(((ds['f_cropprodc_rainfed_spwheat']*ds['area_rainfed_spwheat'])+
                                    (ds['f_cropprodc_irrigated_spwheat']*ds['area_irrigated_spwheat'])+
                                    (ds['f_cropprodc_rainfed_wtwheat']*ds['area_rainfed_wtwheat'])+
                                    (ds['f_cropprodc_irrigated_wtwheat']*ds['area_irrigated_wtwheat']))*
                                    (10**6)*2.5*(10**(-6))/(ds['area_rainfed_spwheat']+ds['area_irrigated_spwheat']+
                                    ds['area_rainfed_wtwheat']+ds['area_irrigated_wtwheat'])*(3600.*24.*365.))/100.
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

      info.sim_varname='Crop_Yield_Wheat'
      info.sim_varunit='t ha-1'
      return info, ds['Crop_Yield_Wheat']

   if info.item == "Canopy_Interception":
      try:
         ds['Canopy_Interception']=ds['f_fevpl']-ds['f_etr']
         info.sim_varname='Canopy_Interception'
         info.sim_varunit=' mm s-1'
      except:
         print('canopy interception evaporation calculation processing ERROR!!!')
      return info, ds['Canopy_Interception']

   if info.item == "Precipitation":
      try:
         if 'Precipitation' in ds.variables:
               # Use method='nearest' to select the nearest value in the 'soil' index
               ds['Precipitation'] = ds['Precipitation']
               info.sim_varname = 'Precipitation'
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
         info.sim_varname='Surface_Net_SW_Radiation'
         info.sim_varunit='W m-2'
      except:
         print('Surface Net SW Radiation calculation processing ERROR!!!')
      return info, ds['Surface_Net_SW_Radiation']
   
   if info.item == "Surface_Net_LW_Radiation":
      try:
         ds['Surface_Net_LW_Radiation']=ds['f_xy_frl']-ds['f_olrg']
         info.sim_varname='Surface_Net_LW_Radiation'
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
      
            info.sim_varname = 'f_wliq_soisno'
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
            info.sim_varname = 'f_wliq_soisno'
            info.sim_varunit = 'unitless'
      except Exception as e:
         print(f"Surface soil moisture calculation processing ERROR: {e}")
         return info, None
      return info, ds['f_wliq_soisno']
   