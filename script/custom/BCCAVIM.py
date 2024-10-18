def filter_BCCAVIM(info, ds):   #update info as well
   if info.item == "Net_Radiation":
      try:
         ds['Net_Radiation'] = ds['FSDS'] - ds['FSR'] + ds['FLDS'] - ds['FIRE']
         info.sim_varname = 'Net_Radiation'
         info.sim_varunit = 'W m-2'
      except Exception as e:
         print(f"Surface Net Radiation calculation processing ERROR: {e}")
         return info, None
      return info, ds['Net_Radiation']
   
   
   if info.item == "Surface_Net_LW_Radiation":
      try:
         ds['FIRA']=-ds['FIRA']
         info.sim_varname='FIRA'
         info.sim_varunit='W m-2'
      except:
         print('Surface Net LW Radiation calculation processing ERROR!!!')
      return info, ds['FIRA']
   
   
   if info.item == "Surface_Soil_Moisture":
      try:
            ds['SOILLIQ']= (ds['SOILLIQ'].isel(levsoi=0) +
                                       ds['SOILLIQ'].isel(levsoi=1))/0.06/1000.0
            info.sim_varname = 'SOILLIQ'
            info.sim_varunit = 'unitless'
      except Exception as e:
         print(f"Surface soil moisture calculation processing ERROR: {e}")
         return info, None
      return info, ds['SOILLIQ']
   
   if info.item == "Root_Zone_Soil_Moisture":
      try:
            ds['SOILLIQ']= (ds['SOILLIQ'].isel(levsoi=0) +
                                       ds['SOILLIQ'].isel(levsoi=1)+
                                       ds['SOILLIQ'].isel(levsoi=2)+
                                       ds['SOILLIQ'].isel(levsoi=3)+
                                       ds['SOILLIQ'].isel(levsoi=4)+
                                       ds['SOILLIQ'].isel(levsoi=5)+
                                       ds['SOILLIQ'].isel(levsoi=6)+
                                       ds['SOILLIQ'].isel(levsoi=7)+
                                       ds['SOILLIQ'].isel(levsoi=8)*0.29
                                       )/1000.0
            info.sim_varname = 'SOILLIQ'
            info.sim_varunit = 'unitless'
      except Exception as e:
         print(f"Surface soil moisture calculation processing ERROR: {e}")
         return info, None
      return info, ds['SOILLIQ']
   
   
   if info.item == "Root_Zone_Soil_Temperature":
      try:
            ds['SOILLIQ']= (ds['SOILLIQ'].isel(levsoi=0) +
                                       ds['SOILLIQ'].isel(levsoi=1)+
                                       ds['SOILLIQ'].isel(levsoi=2)+
                                       ds['SOILLIQ'].isel(levsoi=3)+
                                       ds['SOILLIQ'].isel(levsoi=4)+
                                       ds['SOILLIQ'].isel(levsoi=5)+
                                       ds['SOILLIQ'].isel(levsoi=6)+
                                       ds['SOILLIQ'].isel(levsoi=7)+
                                       ds['SOILLIQ'].isel(levsoi=8)*0.29
                                       )/1000.0
            info.sim_varname = 'SOILLIQ'
            info.sim_varunit = 'unitless'
      except Exception as e:
         print(f"Surface soil moisture calculation processing ERROR: {e}")
         return info, None
      return info, ds['SOILLIQ']
   
 
   if info.item == "Latent_Heat":
      try:
            ds['Latent_Heat']=  ds['FGEV'] + ds['FCEV'] + ds['FCTR']

            info.sim_varname = 'Latent_Heat'
            info.sim_varunit = 'W m-2'
      except Exception as e:
         print(f"Latent Heat calculation processing ERROR: {e}")
         return info, None
      return info, ds['Latent_Heat']
   

   if info.item == "Evapotranspiration":
      try:
            ds['Evapotranspiration']=  (ds['FCEV'] + ds['FCTR']) / 28.4 + ds['QSOIL']  

            info.sim_varname = 'Evapotranspiration'
            info.sim_varunit = 'mm s-1'
      except Exception as e:
         print(f"Evapotranspiration calculation processing ERROR: {e}")
         return info, None
      return info, ds['Evapotranspiration']
   
   
   if info.item == "Canopy_Interception":
      try:
            ds['Canopy_Interception']=  ds['H2OCAN'] / 86400.

            info.sim_varname = 'Canopy_Interception'
            info.sim_varunit = 'mm s-1'
      except Exception as e:
         print(f"Canopy Interception calculation processing ERROR: {e}")
         return info, None
      return info, ds['Canopy_Interception']
   
   
   if info.item == "Canopy_Evaporation_Canopy_Transpiration":
      try:
            ds['Canopy_Evaporation_Canopy_Transpiration']=  ds['FCEV'] + ds['FCTR']

            info.sim_varname = 'Canopy_Evaporation_Canopy_Transpiration'
            info.sim_varunit = 'W m-2'
      except Exception as e:
         print(f"Canopy Evaporation and Transpiration calculation processing ERROR: {e}")
         return info, None
      return info, ds['Canopy_Evaporation_Canopy_Transpiration']
   
   
   if info.item == "Albedo":
      try:
            ds['Albedo']= ds['FSR'] /  ds['FSDS'] 

            info.sim_varname = 'Albedo'
            info.sim_varunit = 'unitless'
      except Exception as e:
         print(f"Surface Albedo calculation processing ERROR: {e}")
         return info, None
      return info, ds['Albedo']

   return info, ds
