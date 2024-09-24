import xarray as xr
class UnitProcessing:
	def __init__(self, info):
		self.name = 'plotting'
		self.version = '0.1'
		self.release = '0.1'
		self.date = 'Mar 2023'
		self.author = "Zhongwang Wei / zhongwang007@gmail.com"
		self.__dict__.update(info)

	def check_units(input_units, target_units):
		"""
		check the consistent of the units

		参数:
			input_units (str): 输入的单位字符串，以空格分隔。
			target_units (str): 目标单位字符串，以空格分隔。

		返回:
			bool: 如果单位一致返回 True，否则返回 False。
		"""

		# 将输入和目标单位字符串拆分为列表，并排序
		input_units_list = sorted(input_units.split())
		target_units_list = sorted(target_units.split())


		# 比较排序后的列表是否相等
		#True: "单位一致"
		#False: "单位不一致"
		return input_units_list == target_units_list

	def process_unit(self, data, unit):
		#GPP unit conversion
				#for metric in self.metrics:
		if hasattr(UnitProcessing, f'Unit_{self.item}'):
			data,unit = getattr(UnitProcessing, f'Unit_{self.item}')(self,data, unit)
		else:
			print(f"Error: The unit of the {self.item} data is not supported!")
			print(f"Please add the unit conversion function for {self.item} in the UnitProcessing class!")
			exit()
		return data,unit
	
  #*******************Ecosystem and Carbon Cycle****************
	def Unit_Gross_Primary_Productivity(self,data,unit):
		standard_units = 'g m-2 day-1'
		print("convert Gross_Primary_Productivity unit to 'g m-2 day-1'")
		print(f"input unit: {unit}")
		print(f"standard unit: {standard_units}")
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list== sorted('mol m-2 s-1'.lower().split()):
				data=data*86400.*12.01
			elif input_units_list== sorted('gc m-2 s-1'.lower().split()):
				data=data*86400.  
			elif input_units_list== sorted('g m-2 s-1'.lower().split()):
				data=data*86400.  
			elif input_units_list== sorted('gc m-2 day-1'.lower().split()):
				data=data
			elif input_units_list== sorted('mumolCO2 m-2 s-1'.lower().split()):
				data=data*12.e-6*86400.
			elif input_units_list== sorted('gC m-2 d-1'.lower().split()):
				pass
			else:
				print(f"Error: The unit of the Gross_Primary_Productivity data is not supported!")
				exit()
		return data,unit
	
	def Unit_Net_Ecosystem_Exchange(self,data,unit):
		standard_units = 'g m-2 day-1'
		print("convert Net_Ecosystem_Exchange unit to 'g m-2 day-1'")
		print(f"input unit: {unit}")
		print(f"standard unit: {standard_units}")
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list== sorted('mol m-2 s-1'.lower().split()):
				data=data*86400.*12.01
			elif input_units_list== sorted('gc m-2 s-1'.lower().split()):
				data=data*86400.  
			elif input_units_list== sorted('g m-2 s-1'.lower().split()):
				data=data*86400.  
			elif input_units_list== sorted('gc m-2 day-1'.lower().split()):
				data=data
			elif input_units_list== sorted('mumolCO2 m-2 s-1'.lower().split()):
				data=data*12.e-6*86400.
			elif input_units_list== sorted('gC m-2 d-1'.lower().split()):
				pass
			else:
				print(f"Error: The unit of the Net_Ecosystem_Exchange data is not supported!")
				exit()
		return data,unit

	def Unit_Ecosystem_Respiration(self,data,unit):
		standard_units = 'g m-2 day-1'
		print("convert Ecosystem_Respiration unit to 'g m-2 day-1'")
		print(f"input unit: {unit}")
		print(f"standard unit: {standard_units}")
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list== sorted('mol m-2 s-1'.lower().split()):
				data=data*86400.*12.01
			elif input_units_list== sorted('gc m-2 s-1'.lower().split()):
				data=data*86400.  
			elif input_units_list== sorted('g m-2 s-1'.lower().split()):
				data=data*86400.  
			elif input_units_list== sorted('gc m-2 day-1'.lower().split()):
				data=data
			elif input_units_list== sorted('mumolCO2 m-2 s-1'.lower().split()):
				data=data*12.e-6*86400.
			elif input_units_list== sorted('gC m-2 d-1'.lower().split()):
				pass
			else:
				print(f"Error: The unit of the Ecosystem_Respiration data is not supported!")
				exit()
		return data,unit

	def Unit_Biomass(self,data,unit):
		standard_units = 'g m-2'
		print("convert Biomass unit to 'g m-2'")
		print(f"input unit: {unit}")
		print(f"standard unit: {standard_units}")
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list== sorted('kg m-2'.lower().split()):
				data=data*1000.
			elif input_units_list== sorted('g m-2'.lower().split()):
				pass
			elif input_units_list== sorted('Mg ha-1'.lower().split()):
				data=100*data
			else:
				print(f"Error: The unit of the Biomass data is not supported!")
				exit()
		return data,unit

	def Unit_Burned_Area(self,data,unit):
		standard_units = 'fraction'
		print("convert Burned_Area unit to 'fraction'")
		print(f"input unit: {unit}")
		print(f"standard unit: {standard_units}")
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list== sorted('m2 m-2'.lower().split()):
				pass
			elif input_units_list== sorted('fraction'.lower().split()):
				pass
			elif input_units_list== sorted('percent'.lower().split()):
				data=data/100.
			elif input_units_list== sorted('percentage'.lower().split()):
				data=data/100.
			else:
				print(f"Error: The unit of the Burned_Area data is not supported!")
				exit()
		return data,unit

	def Unit_Nitrogen_Fixation(self,data,unit):
		###need check
		standard_units = 'g m-2 day-1'
		print("convert Nitrogen_Fixation unit to 'g m-2 day-1'")
		print(f"input unit: {unit}")
		print(f"standard unit: {standard_units}")
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list== sorted('kg m-2 day-1'.lower().split()):
				data=data*1000.
			elif input_units_list== sorted('g m-2 day-1'.lower().split()):
				pass
			elif input_units_list== sorted('g m-2 s-1'.lower().split()):
				data=data*86400.
			elif input_units_list== sorted('g m-2 hr-1'.lower().split()):
				data=data*86400.*3600.
			else:
				print(f"Error: The unit of the Nitrogen_Fixation data is not supported!")
				exit()
		return data,unit

	def Unit_Soil_Carbon(self,data,unit):
		standard_units = 'g m-2'
		print("convert Soil_Carbon unit to 'g m-2'")
		print(f"input unit: {unit}")
		print(f"standard unit: {standard_units}")
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list== sorted('kg m-2'.lower().split()):
				data=data*1000.
			elif input_units_list== sorted('g m-2'.lower().split()):
				pass
			elif input_units_list== sorted('Mg ha-1'.lower().split()):
				data=100*data
			else:
				print(f"Error: The unit of the Soil_Carbon data is not supported!")
				exit()
		return data,unit

	def Unit_Methane(self,data,unit):
		standard_units = 'g m-2 day-1'
		print("convert Methane unit to 'g m-2 day-1'")
		print(f"input unit: {unit}")
		print(f"standard unit: {standard_units}")
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list== sorted('kg m-2 day-1'.lower().split()):
				data=data*1000.
			elif input_units_list== sorted('g m-2 day-1'.lower().split()):
				pass
			else:
				print(f"Error: The unit of the Methane data is not supported!")
				exit()
		return data,unit

	def Unit_Leaf_Area_Index(self,data,unit):
		standard_units = 'unitless'
		print("convert Leaf_Area_Index unit to 'unitless'")
		print(f"input unit: {unit}")
		print(f"standard unit: {standard_units}")
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list== sorted('m2 m-2'.lower().split()):
				pass
			elif input_units_list== sorted('unitless'.lower().split()):
				pass
			else:
				print(f"Error: The unit of the Leaf_Area_Index data is not supported!")
				exit()
		return data,unit
#-----------------------------------------------------------------------------------
  #****************************      Hydrology Cycle      **********************************
	def Unit_Evapotranspiration(self,data, unit):
		#convert to 'mm s-1'
		standard_units = 'mm day-1'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('mm day-1'.lower().split()):
				pass
			elif input_units_list==sorted('kg m-2 s-1'.lower().split()):
				data=data*86400.
			elif input_units_list==sorted('mm s-1'.lower().split()):
				data=data*86400.
			elif input_units_list==sorted('mm hr-1'.lower().split()):
				data=data*24.
			elif input_units_list==sorted('mm mon-1'.lower().split()):
				data=data/30.0
			elif input_units_list==sorted('mm m-1'.lower().split()):
				data=data/30.0
			elif input_units_list==sorted('mm month-1'.lower().split()):
				data=data/30.0
			else:
				print(f"Error: The unit of the Evapotranspiration data is not supported!")
				exit()
		return data,unit
	
	def Unit_Canopy_Transpiration(self,data, unit):
		#convert to 'mm s-1'
		standard_units = 'mm day-1'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('mm day-1'.lower().split()):
				pass
			elif input_units_list==sorted('kg m-2 s-1'.lower().split()):
				data=data*86400.
			elif input_units_list==sorted('mm s-1'.lower().split()):
				data=data*86400.
			elif input_units_list==sorted('mm hr-1'.lower().split()):
				data=data*24.
			else:
				print(f"Error: The unit of the Transpiration data is not supported!")
				exit()
		return data,unit
	
	def Unit_Canopy_Interception(self,data, unit):
		#convert to 'mm s-1'
		standard_units = 'mm day-1'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('mm day-1'.lower().split()):
				pass
			elif input_units_list==sorted('mm month-1'.lower().split()):
				data=data/30.
			elif input_units_list==sorted('mm m-1'.lower().split()):
				data=data/30.
			elif input_units_list==sorted('mm mon-1'.lower().split()):
				data=data/30.
			elif input_units_list==sorted('mm s-1'.lower().split()):
				data=data*86400.
			else:
				print(f"Error: The unit of the inteception data is not supported!")
				exit()
		return data,unit

	def Unit_Soil_Evaporation(self,data, unit):
		#convert to 'mm s-1'
		standard_units = 'mm day-1'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('mm day-1'.lower().split()):
				pass
			elif input_units_list==sorted('mm month-1'.lower().split()):
				data=data/30.
			elif input_units_list==sorted('mm m-1'.lower().split()):
				data=data/30.
			elif input_units_list==sorted('mm mon-1'.lower().split()):
				data=data/30.
			elif input_units_list==sorted('mm s-1'.lower().split()):
				data=data*86400.
			else:
				print(f"Error: The unit of the Soil_Evaporation data is not supported!")
				exit()
		return data,unit

	def Unit_Open_Water_Evaporation(self,data, unit):
		#convert to 'mm s-1'
		standard_units = 'mm day-1'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('mm day-1'.lower().split()):
				pass
			elif input_units_list==sorted('mm month-1'.lower().split()):
				data=data/30.
			elif input_units_list==sorted('mm m-1'.lower().split()):
				data=data/30.
			elif input_units_list==sorted('mm mon-1'.lower().split()):
				data=data/30.
			elif input_units_list==sorted('mm s-1'.lower().split()):
				data=data*86400.
			else:
				print(f"Error: The unit of the Soil_Evaporation data is not supported!")
				exit()
		return data,unit

	def Unit_Total_Runoff(self,data, unit):
		#convert to 'mm day-1'
		standard_units = 'mm day-1'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('mm s-1'.lower().split()):
				data=data*86400.
			elif input_units_list==sorted('kg m-2 s-1'.lower().split()):
				data=data*86400.
			elif input_units_list==sorted('m day-1'.lower().split()):
				data=data/1000.
			else:
				print(f"Error: The unit of the Runoff data is not supported!")
				exit()
		return data,unit
	
	def Unit_Root_Zone_Soil_Moisture(self,data, unit):
		standard_units = 'unitless'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			print(f"Error: The unit of the Root_Zone_Soil_Moisture data is not supported!")
			exit()
		return data,unit

	def Unit_Surface_Soil_Moisture(self,data, unit):
		standard_units = 'unitless'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('m3 m-3'.lower().split()):
				pass
			elif input_units_list==sorted('unitless'.lower().split()):
				pass
			else:
				print(f"Error: The unit of the Surface_Soil_Moisture data is not supported!")
				exit()
		return data,unit

#-----------------------------------------------------------------------------------
#*******************  Radiation and Energy Cycle  *************
	def Unit_Net_Radiation(self,data, unit):
		#convert to 'w m-2'
		standard_units = 'w m-2'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('MJ m-2 day-1'.lower().split()):
				data=data/0.0864
			elif input_units_list==sorted('W m-2'.lower().split()):
				pass
			elif input_units_list==sorted('MJ m-2 d-1'.lower().split()):
				data=data/0.0864
			else:
				print(f"Error: The unit of the Latent_Heat data is not supported!")
				exit()
		return data,unit

	def Unit_Latent_Heat(self,data, unit):
		#convert to 'w m-2'
		standard_units = 'w m-2'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('MJ m-2 day-1'.lower().split()):
				data=data/0.0864
			elif input_units_list==sorted('W m-2'.lower().split()):
				pass
			elif input_units_list==sorted('MJ m-2 d-1'.lower().split()):
				data=data/0.0864
			else:
				print(f"Error: The unit of the Latent_Heat data is not supported!")
				exit()
		return data,unit

	def Unit_Sensible_Heat(self,data, unit):
		#convert to 'w m-2'
		standard_units = 'w m-2'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('MJ m-2 day-1'.lower().split()):
				data=data/0.086401
			elif input_units_list==sorted('W m-2'.lower().split()):
				pass
			elif input_units_list==sorted('MJ m-2 d-1'.lower().split()): 
				data=data/0.086401
			else:
				print(f"Error: The unit of the Latent_Heat data is not supported!")
				exit()
		return data,unit

	def Unit_Ground_Heat(self,data, unit):
		#convert to 'w m-2'
		standard_units = 'w m-2'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('MJ m-2 day-1'.lower().split()):
				data=data/0.0864
			else:
				print(f"Error: The unit of the Sensible_Heat data is not supported!")
				exit()
		return data,unit

	def Unit_Surface_Upward_SW_Radiation(self,data, unit):
		#convert to 'w m-2'
		standard_units = 'w m-2'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('MJ m-2 day-1'.lower().split()):
				data=data/0.0864
			else:
				print(f"Error: The unit of the Sensible_Heat data is not supported!")
				exit()
		return data,unit

	def Unit_Surface_Upward_LW_Radiation(self,data, unit):
		#convert to 'w m-2'
		standard_units = 'w m-2'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('MJ m-2 day-1'.lower().split()):
				data=data/0.0864
			else:
				print(f"Error: The unit of the Sensible_Heat data is not supported!")
				exit()
		return data,unit
	
	def Unit_Surface_Net_SW_Radiation(self,data, unit):
		#convert to 'w m-2'
		standard_units = 'w m-2'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('MJ m-2 day-1'.lower().split()):
				data=data/0.0864
			else:
				print(f"Error: The unit of the Sensible_Heat data is not supported!")
				exit()
		return data,unit
	
	def Unit_Surface_Net_LW_Radiation(self,data, unit):
		#convert to 'w m-2'
		standard_units = 'w m-2'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('MJ m-2 day-1'.lower().split()):
				data=data/0.0864
			else:
				print(f"Error: The unit of the Sensible_Heat data is not supported!")
				exit()
		return data,unit
	
	def Unit_Albedo(self,data, unit):
		#convert to 'unitless'
		standard_units = 'unitless'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			print(f"Error: The unit of the Albedo data is not supported!")
			exit()
		return data,unit

  #****************************      Forcing      **********************************
	def Unit_Surface_Air_Temperature(self,data, unit):
		#convert to 'K'
		standard_units = 'K'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('C'.lower().split()):
				data=data+273.15
			else:
				print(f"Error: The unit of the Surface_Air_Temperature data is not supported!")
				exit()
		return data,unit

	def Unit_Diurnal_Max_Temperature(self,data, unit):
		#convert to 'K'
		standard_units = 'K'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('C'.lower().split()):
				data=data+273.15
			else:
				print(f"Error: The unit of the Diurnal_Max_Temperature data is not supported!")
				exit()
		return data,unit

	def Unit_Diurnal_Min_Temperature(self,data, unit):
		#convert to 'K'
		standard_units = 'K'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('C'.lower().split()):
				data=data+273.15
			else:
				print(f"Error: The unit of the Diurnal_Min_Temperature data is not supported!")
				exit()
		return data,unit

	def Unit_Diurnal_Temperature_Range(self,data, unit):
		#convert to 'K'
		standard_units = 'K'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('C'.lower().split()):
				data=data+273.15
			else:
				print(f"Error: The unit of the Diurnal_Temperature_Range data is not supported!")
				exit()
		return data,unit

	def Unit_Surface_Downward_SW_Radiation(self,data, unit):
		#convert to 'w m-2'
		standard_units = 'w m-2'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('MJ m-2 day-1'.lower().split()):
				data=data*0.0864
			else:
				print(f"Error: The unit of the Surface_Downward_SW_Radiation data is not supported!")
				exit()
		return data,unit

	def Unit_Surface_Downward_LW_Radiation(self,data, unit):
		#convert to 'w m-2'
		standard_units = 'w m-2'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('MJ m-2 day-1'.lower().split()):
				data=data*0.0864
			else:
				print(f"Error: The unit of the Surface_Downward_LW_Radiation data is not supported!")
				exit()
		return data,unit

	def Unit_Surface_Relative_Humidity(self,data, unit):
		standard_units = 'unitless'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('%'.lower().split()):
				data=data/100.
			else:
				print(f"Error: The unit of the Surface_Relative_Humidity data is not supported!")
				exit()
		return data,unit

	def Unit_Surface_Specific_Humidity(self,data, unit):
		standard_units = 'unitless'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('g kg-1'.lower().split()):
				data=data/1000.
			else:
				print(f"Error: The unit of the Surface_Specific_Humidity data is not supported!")
				exit()
		return data,unit

	def Unit_Precipitation(self,data, unit):
		standard_units = 'mm day-1'
		input_units_list  = sorted(unit.lower().split())
		print(input_units_list)
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('mm s-1'.lower().split()):
				data=data*86400.
			elif input_units_list==sorted('kg m-2 s-1'.lower().split()):
				data=data*86400.
			elif input_units_list==sorted('mm hr-1'.lower().split()):
				data=data * 24
			elif input_units_list==sorted('mm day-1'.lower().split()):
				pass	
			elif input_units_list==sorted('mm d-1'.lower().split()):
				pass
			elif input_units_list==sorted('mm month-1'.lower().split()):
				data=data/ 30.
			elif input_units_list==sorted('mm mon-1'.lower().split()):
				data=data/ 30.
			elif input_units_list==sorted('mm m-1'.lower().split()):
				data=data/ 30.
			else:
				print(f"Error: The unit of the Precepitation data is not supported!")
				exit()
		return data,unit	




#*******************    Human activity       ***************
#---------------------------Urban---------------------------
	def Unit_Urban_Anthropogenic_Heat_Flux(self,data,unit):
		standard_units = 'w m-2'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			print(f"Error: The unit of the Ground_Heat_Flux data is not supported!")
			exit()
		return data,unit

	def Unit_Urban_Albedo(self,data,unit):
		standard_units = 'unitless'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			print(f"Error: The unit of the Ground_Heat_Flux data is not supported!")
			exit()
		return data,unit

	def Unit_Urban_Latent_Heat_Flux(self,data,unit):
		#convert to 'w m-2'
		standard_units = 'w m-2'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			print(f"Error: The unit of the Sensible_Heat data is not supported!")
			exit()
		return data,unit

	def Unit_Urban_Surface_Temperature(self,data,unit):
		#convert to 'w m-2'
		standard_units = 'K'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			print(f"Error: The unit of the Sensible_Heat data is not supported!")
			exit()
		return data,unit

	def Unit_Urban_Air_Temperature(self,data,unit):
		#convert to 'w m-2'
		standard_units = 'K'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			print(f"Error: The unit of the Sensible_Heat data is not supported!")
			exit()
		return data,unit

#---------------------------Dam---------------------------
	def Unit_Dam_Inflow(self,data,unit):
		standard_units = 'm3 s-1'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('m3 s-1'.lower().split()):
				pass
			elif input_units_list==sorted('m3 day-1'.lower().split()):
				data=data/86400.
			else:
				print(f"Error: The unit of the Dam_Inflow data is not supported!")
				exit()
		return data,unit

	def Unit_Dam_Outflow(self,data,unit):
		standard_units = 'm3 s-1'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('m3 s-1'.lower().split()):
				pass
			elif input_units_list==sorted('m3 day-1'.lower().split()):
				data=data/86400.
			else:
				print(f"Error: The unit of the Dam_Outflow data is not supported!")
				exit()
		return data,unit
	
	def Unit_Dam_Water_Level(self,data,unit):
		standard_units = 'm'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('m'.lower().split()):
				pass
			elif input_units_list==sorted('cm'.lower().split()):
				data=data/100.
			else:
				print(f"Error: The unit of the Dam_Water_Level data is not supported!")
				exit()
		return data,unit
	
	def Unit_Dam_Water_Storage(self,data,unit):
		standard_units = 'mcm'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('mcm'.lower().split()):
				pass
			elif input_units_list==sorted('m3'.lower().split()):
				data=data/1.e6
			elif input_units_list==sorted('million cubic meters'.lower().split()):
				pass
			else:
				print(f"Error: The unit of the Dam_Water_Storage data is not supported!")
				exit()
		return data,unit

#---------------------------River---------------------------
	def Unit_StreamFlow(self,data, unit):
		#convert to 'm3 s-1'
		standard_units = 'm3 s-1'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('m3 Day-1'.lower().split()):
				data=data/86400.
			else:
				print(f"Error: The unit of the StreamFlow data is not supported!")
				exit()
		return data,unit

	def Unit_River_Water_Level(self,data, unit):
		#convert to 'm'
		standard_units = 'm'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('m'.lower().split()):
				pass
			elif input_units_list==sorted('cm'.lower().split()):
				data=data/100.
			else:
				print(f"Error: The unit of the River_Water_Level data is not supported!")
				exit()
		return data,unit

	def Unit_Inundation_Area(self,data, unit):
		#convert to 'm2'
		standard_units = 'km2'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('km2'.lower().split()):
				pass
			elif input_units_list==sorted('m2'.lower().split()):
				data=data/1.e6
			else:
				print(f"Error: The unit of the Inundation_Area data is not supported!")
				exit()
		return data,unit
	
	def Unit_Inundation_Fraction(self, data, unit):
		#convert to 'fraction'
		standard_units = 'unitless'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		else:
			unit = standard_units
			if input_units_list==sorted('fraction'.lower().split()):
				pass
			elif input_units_list==sorted('percent'.lower().split()):
				data=data/100.
			elif input_units_list==sorted('percentage'.lower().split()):
				data=data/100.
			else:
				print(f"Error: The unit of the Inundation_Fraction data is not supported!")
				exit()
		return data,unit
#---------------------------Lake---------------------------


#---------------------------Crop---------------------------
	def Unit_Total_Irrigation_Amount(self,data, unit):
		#convert to 'kg m-2'
		standard_units = 'mm year-1'
		input_units_list  = sorted(unit.lower().split())
		target_units_list = sorted(standard_units.lower().split())
		if input_units_list == target_units_list:
			pass
		elif input_units_list==sorted('kg m-2'.lower().split()):
			pass  
		elif input_units_list==sorted('mm month-1'.lower().split()):
			data=data*12.
		elif input_units_list==sorted('mm day-1'.lower().split()):
			data=data*365.
		elif input_units_list==sorted('mm s-1'.lower().split()):
			data=data*365.*86400.
		else:
			print(f"Error: The unit of the Total_Irrigation_Amount data is not supported!")
			exit()
		return data,unit