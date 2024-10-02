def filter_FLUXCOM(info,ds):   #update info as well
   if info.item == "Net_Radiation":
      try:
         ds['Rn']=ds['Rn']
      except:
         print('Surface Net SW Radiation calculation processing ERROR!!!')
      return info, ds['Rn']
   
   if info.item == "Latent_Heat":
      try:
         ds['LE']=ds['LE']
      except:
         print('Latent Heat calculation processing ERROR!!!')
      return info, ds['LE']
   if info.item == "Sensible_Heat":
      try:
         ds['H']=ds['H']
      except:
         print('Sensible Heat calculation processing ERROR!!!')
      return info, ds['H']
   