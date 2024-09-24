def filter_CLM5(info,ds):   #update info as well
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