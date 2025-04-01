def filter_SMAP_L4(info,ds):   
   if info.item == "Net_Radiation":
      try:
         ds['Net_Radiation'] = ds['net_downward_longwave_flux'] + ds['net_downward_shortwave_flux'] 
         info.sim_varname = 'Net_Radiation'
         info.sim_varunit = 'W m-2'
      except Exception as e:
         print(f"Surface Net Radiation calculation processing ERROR: {e}")
         return info, None
      return info, ds['Net_Radiation']
