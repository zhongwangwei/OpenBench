def filter_GLDAS2(info,ds):   #update info as well
   if info.item == "Net_Radiation":
      try:
         ds['Net_Radiation']=ds['Swnet_tavg'] + ds['Lwnet_tavg']
         info.sim_varname='Net_Radiation'
         info.sim_varunit='W m-2'
      except:
         print('Surface Net_Radiation calculation processing ERROR!!!')
      return info, ds['Net_Radiation']
   
   if info.item == "Surface_Upward_SW_Radiation":
      try:
         ds['Surface_Upward_SW_Radiation']=ds['SWdown_f_tavg']-ds['Swnet_tavg']
         info.sim_varname='Surface_Upward_SW_Radiation'
         info.sim_varunit='W m-2'
      except:
         print('Surface_Upward_SW_Radiation calculation processing ERROR!!!')
      return info, ds['Surface_Upward_SW_Radiation']
   
   if info.item == "Surface_Upward_LW_Radiation":
      try:
         ds['Surface_Upward_LW_Radiation']=ds['LWdown_f_tavg']-ds['Lwnet_tavg']
         info.sim_varname='Surface_Upward_LW_Radiation'
         info.sim_varunit='W m-2'
      except:
         print('Surface_Upward_LW_Radiation calculation processing ERROR!!!')
      return info, ds['Surface_Upward_LW_Radiation']
                       
