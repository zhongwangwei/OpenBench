# -*- coding: utf-8 -*-
import glob, shutil
import os
import sys
from pathlib import Path
import itertools
import platform
import posixpath
from posixpath import normpath
import time
import streamlit as st
from PIL import Image
from io import StringIO
from collections import ChainMap
import xarray as xr
import numpy as np
from Namelist_lib.namelist_read import NamelistReader
from Namelist_lib.namelist_info import initial_setting
from Namelist_lib.find_path import FindPath


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = (time_end - time_start) / 60.
        print('%s cost time: %.3f min' % (func.__name__, time_spend))
        return result

    return func_wrapper


class mange_simulation:
    def __init__(self):
        self.author = "Qingchen Xu/xuqingchen0@gmail.com"

    def step3_make_new_simnml(self):
        if 'step3_add_nml' not in st.session_state:
            st.session_state.step3_add_nml = True

        sim_save_path = self.__check_path()
        st.divider()

        def variables_change(key, editor_key):
            newlib[key] = st.session_state[editor_key]

        def sim_main_change(key, editor_key):
            newlib['general'][key] = st.session_state[editor_key]

        newlib = {}
        col1, col2, col3 = st.columns((1.2, 1.5, 1.5))
        col1.write('###### :green[Simulation case name]')
        col2.write('###### :green[Mod variables selected]')
        col3.write('###### :point_down: :green[press to add new model]')

        col1, col2, col3, col4 = st.columns((1, 1.5, 0.8, 0.8))
        newlib['Sim_casename'] = col1.text_input(f'Simulation case name: ', value='',
                                                 key=f"Sim_casename",
                                                 on_change=variables_change,
                                                 args=(f"Sim_casename", 'Sim_casename'),
                                                 label_visibility='collapsed',
                                                 placeholder=f"Simulation case...")
        newlib['Mod'] = col2.selectbox("Mod variables selected", sorted(self.sim_sources['def_Mod'].keys()), index=None,
                                       key=f"Mod_select",
                                       on_change=variables_change,
                                       args=('Mod', f"Mod_select"),
                                       placeholder="Choose an option",
                                       label_visibility='collapsed')

        def define_new_mod():
            st.session_state.add_mod = True

        def define_finish():
            st.session_state.add_mod = False
            del_vars = ['mode_item']
            for del_var in del_vars:
                del st.session_state[del_var]
            self.sim_sources = self.nl.read_namelist('./GUI/Namelist_lib/Simulation_lib.nml')

        col3.button('Add new Mod', on_click=define_new_mod)
        col4.button('Finish add', on_click=define_finish)

        def get_var(col, name):
            if name not in st.session_state:
                st.session_state[name] = self.initial.evaluation_items()
            Evaluation_Items = st.session_state[name]
            col.write('')
            with col.popover("Variables items", use_container_width=True):
                def Evaluation_Items_editor_change(key, editor_key):
                    Evaluation_Items[key] = st.session_state[key]

                st.subheader("Mod Variables ....", divider=True)
                st.write('##### :blue[Ecosystem and Carbon Cycle]')
                # st.subheader("", divider=True, )
                st.checkbox("Gross Primary Productivity", key="Gross_Primary_Productivity",
                            on_change=Evaluation_Items_editor_change,
                            args=("Gross_Primary_Productivity", "Gross_Primary_Productivity"),
                            value=Evaluation_Items["Gross_Primary_Productivity"])
                st.checkbox("Ecosystem Respiration", key="Ecosystem_Respiration",
                            on_change=Evaluation_Items_editor_change,
                            args=("Ecosystem_Respiration", "Ecosystem_Respiration"),
                            value=Evaluation_Items["Ecosystem_Respiration"])
                st.checkbox("Net Ecosystem Exchange", key="Net_Ecosystem_Exchange",
                            on_change=Evaluation_Items_editor_change,
                            args=("Net_Ecosystem_Exchange", "Net_Ecosystem_Exchange"),
                            value=Evaluation_Items["Net_Ecosystem_Exchange"])
                st.checkbox("Leaf Area Index", key="Leaf_Area_Index", on_change=Evaluation_Items_editor_change,
                            args=("Leaf_Area_Index", "Leaf_Area_Index"), value=Evaluation_Items["Leaf_Area_Index"])
                st.checkbox("Biomass", key="Biomass", on_change=Evaluation_Items_editor_change,
                            args=("Biomass", "Biomass"),
                            value=Evaluation_Items["Biomass"])
                st.checkbox("Burned Area", key="Burned_Area", on_change=Evaluation_Items_editor_change,
                            args=("Burned_Area", "Burned_Area"), value=Evaluation_Items["Burned_Area"])
                st.checkbox("Soil Carbon", key="Soil_Carbon", on_change=Evaluation_Items_editor_change,
                            args=("Soil_Carbon", "Soil_Carbon"), value=Evaluation_Items["Soil_Carbon"])
                st.checkbox("Nitrogen Fixation", key="Nitrogen_Fixation", on_change=Evaluation_Items_editor_change,
                            args=("Nitrogen_Fixation", "Nitrogen_Fixation"),
                            value=Evaluation_Items["Nitrogen_Fixation"])
                st.checkbox("Methane", key="Methane", on_change=Evaluation_Items_editor_change,
                            args=("Methane", "Methane"),
                            value=Evaluation_Items["Methane"])
                st.checkbox("Veg Cover In Fraction", key="Veg_Cover_In_Fraction",
                            on_change=Evaluation_Items_editor_change,
                            args=("Veg_Cover_In_Fraction", "Veg_Cover_In_Fraction"),
                            value=Evaluation_Items["Veg_Cover_In_Fraction"])
                st.checkbox("Leaf Greenness", key="Leaf_Greenness", on_change=Evaluation_Items_editor_change,
                            args=("Leaf_Greenness", "Leaf_Greenness"), value=Evaluation_Items["Leaf_Greenness"])

                st.write('##### :blue[Radiation and Energy Cycle]')
                st.checkbox("Net Radiation", key="Net_Radiation", on_change=Evaluation_Items_editor_change,
                            args=("Net_Radiation", "Net_Radiation"), value=Evaluation_Items["Net_Radiation"])
                st.checkbox("Latent Heat", key="Latent_Heat", on_change=Evaluation_Items_editor_change,
                            args=("Latent_Heat", "Latent_Heat"), value=Evaluation_Items["Latent_Heat"])
                st.checkbox("Sensible Heat", key="Sensible_Heat", on_change=Evaluation_Items_editor_change,
                            args=("Sensible_Heat", "Sensible_Heat"), value=Evaluation_Items["Sensible_Heat"])
                st.checkbox("Ground Heat", key="Ground_Heat", on_change=Evaluation_Items_editor_change,
                            args=("Ground_Heat", "Ground_Heat"), value=Evaluation_Items["Ground_Heat"])
                st.checkbox("Albedo", key="Albedo", on_change=Evaluation_Items_editor_change, args=("Albedo", "Albedo"),
                            value=Evaluation_Items["Albedo"])
                st.checkbox("Surface Upward SW Radiation", key="Surface_Upward_SW_Radiation",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Upward_SW_Radiation", "Surface_Upward_SW_Radiation"),
                            value=Evaluation_Items["Surface_Upward_SW_Radiation"])
                st.checkbox("Surface Upward LW Radiation", key="Surface_Upward_LW_Radiation",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Upward_LW_Radiation", "Surface_Upward_LW_Radiation"),
                            value=Evaluation_Items["Surface_Upward_LW_Radiation"])
                st.checkbox("Surface Net SW Radiation", key="Surface_Net_SW_Radiation",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Net_SW_Radiation", "Surface_Net_SW_Radiation"),
                            value=Evaluation_Items["Surface_Net_SW_Radiation"])
                st.checkbox("Surface Net LW Radiation", key="Surface_Net_LW_Radiation",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Net_LW_Radiation", "Surface_Net_LW_Radiation"),
                            value=Evaluation_Items["Surface_Net_LW_Radiation"])
                st.checkbox("Surface Soil Temperature", key="Surface_Soil_Temperature",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Soil_Temperature", "Surface_Soil_Temperature"),
                            value=Evaluation_Items["Surface_Soil_Temperature"])
                st.checkbox("Root Zone Soil Temperature", key="Root_Zone_Soil_Temperature",
                            on_change=Evaluation_Items_editor_change,
                            args=("Root_Zone_Soil_Temperature", "Root_Zone_Soil_Temperature"),
                            value=Evaluation_Items["Root_Zone_Soil_Temperature"])

                st.write('##### :blue[Forcings]')
                # st.subheader(":blue[]", divider=True)
                st.checkbox("Diurnal Temperature Range", key="Diurnal_Temperature_Range",
                            on_change=Evaluation_Items_editor_change,
                            args=("Diurnal_Temperature_Range", "Diurnal_Temperature_Range"),
                            value=Evaluation_Items["Diurnal_Temperature_Range"])
                st.checkbox("Diurnal Max Temperature", key="Diurnal_Max_Temperature",
                            on_change=Evaluation_Items_editor_change,
                            args=("Diurnal_Max_Temperature", "Diurnal_Max_Temperature"),
                            value=Evaluation_Items["Diurnal_Max_Temperature"])
                st.checkbox("Diurnal Min Temperature", key="Diurnal_Min_Temperature",
                            on_change=Evaluation_Items_editor_change,
                            args=("Diurnal_Min_Temperature", "Diurnal_Min_Temperature"),
                            value=Evaluation_Items["Diurnal_Min_Temperature"])
                st.checkbox("Surface Downward SW Radiation", key="Surface_Downward_SW_Radiation",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Downward_SW_Radiation", "Surface_Downward_SW_Radiation"),
                            value=Evaluation_Items["Surface_Downward_SW_Radiation"])
                st.checkbox("Surface Downward LW Radiation", key="Surface_Downward_LW_Radiation",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Downward_LW_Radiation", "Surface_Downward_LW_Radiation"),
                            value=Evaluation_Items["Surface_Downward_LW_Radiation"])
                st.checkbox("Surface Relative Humidity", key="Surface_Relative_Humidity",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Relative_Humidity", "Surface_Relative_Humidity"),
                            value=Evaluation_Items["Surface_Relative_Humidity"])
                st.checkbox("Surface Specific Humidity", key="Surface_Specific_Humidity",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Specific_Humidity", "Surface_Specific_Humidity"),
                            value=Evaluation_Items["Surface_Specific_Humidity"])
                st.checkbox("Precipitation", key="Precipitation", on_change=Evaluation_Items_editor_change,
                            args=("Precipitation", "Precipitation"), value=Evaluation_Items["Precipitation"])
                st.checkbox("Surface Air Temperature", key="Surface_Air_Temperature",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Air_Temperature", "Surface_Air_Temperature"),
                            value=Evaluation_Items["Surface_Air_Temperature"])

                st.write('##### :blue[Hydrology Cycle]')
                # st.subheader(":blue[]", divider=True)

                st.checkbox("Evapotranspiration", key="Evapotranspiration", on_change=Evaluation_Items_editor_change,
                            args=("Evapotranspiration", "Evapotranspiration"),
                            value=Evaluation_Items["Evapotranspiration"])
                st.checkbox("Canopy Transpiration", key="Canopy_Transpiration",
                            on_change=Evaluation_Items_editor_change,
                            args=("Canopy_Transpiration", "Canopy_Transpiration"),
                            value=Evaluation_Items["Canopy_Transpiration"])
                st.checkbox("Canopy Interception", key="Canopy_Interception",
                            on_change=Evaluation_Items_editor_change,
                            args=("Canopy_Interception", "Canopy_Interception"),
                            value=Evaluation_Items["Canopy_Interception"])
                st.checkbox("Ground Evaporation", key="Ground_Evaporation", on_change=Evaluation_Items_editor_change,
                            args=("Ground_Evaporation", "Ground_Evaporation"),
                            value=Evaluation_Items["Ground_Evaporation"])
                st.checkbox("Water Evaporation", key="Water_Evaporation", on_change=Evaluation_Items_editor_change,
                            args=("Water_Evaporation", "Water_Evaporation"),
                            value=Evaluation_Items["Water_Evaporation"])
                st.checkbox("Soil Evaporation", key="Soil_Evaporation", on_change=Evaluation_Items_editor_change,
                            args=("Soil_Evaporation", "Soil_Evaporation"), value=Evaluation_Items["Soil_Evaporation"])
                st.checkbox("Total Runoff", key="Total_Runoff", on_change=Evaluation_Items_editor_change,
                            args=("Total_Runoff", "Total_Runoff"), value=Evaluation_Items["Total_Runoff"])
                st.checkbox("Terrestrial Water Storage Change", key="Terrestrial_Water_Storage_Change",
                            on_change=Evaluation_Items_editor_change,
                            args=("Terrestrial_Water_Storage_Change", "Terrestrial_Water_Storage_Change"),
                            value=Evaluation_Items["Terrestrial_Water_Storage_Change"])
                st.checkbox("Snow Water Equivalent", key="Snow_Water_Equivalent",
                            on_change=Evaluation_Items_editor_change,
                            args=("Snow_Water_Equivalent", "Snow_Water_Equivalent"),
                            value=Evaluation_Items["Snow_Water_Equivalent"])

                st.checkbox("Surface Snow Cover In Fraction", key="Surface_Snow_Cover_In_Fraction",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Snow_Cover_In_Fraction", "Surface_Snow_Cover_In_Fraction"),
                            value=Evaluation_Items["Surface_Snow_Cover_In_Fraction"])
                st.checkbox("Snow Depth", key="Snow_Depth", on_change=Evaluation_Items_editor_change,
                            args=("Snow_Depth", "Snow_Depth"), value=Evaluation_Items["Snow_Depth"])
                st.checkbox("Permafrost", key="Permafrost", on_change=Evaluation_Items_editor_change,
                            args=("Permafrost", "Permafrost"), value=Evaluation_Items["Permafrost"])
                st.checkbox("Surface Soil Moisture", key="Surface_Soil_Moisture",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Soil_Moisture", "Surface_Soil_Moisture"),
                            value=Evaluation_Items["Surface_Soil_Moisture"])
                st.checkbox("Root Zone Soil Moisture", key="Root_Zone_Soil_Moisture",
                            on_change=Evaluation_Items_editor_change,
                            args=("Root_Zone_Soil_Moisture", "Root_Zone_Soil_Moisture"),
                            value=Evaluation_Items["Root_Zone_Soil_Moisture"])
                st.checkbox("Water Table Depth", key="Water_Table_Depth", on_change=Evaluation_Items_editor_change,
                            args=("Water_Table_Depth", "Water_Table_Depth"),
                            value=Evaluation_Items["Water_Table_Depth"])
                st.checkbox("Water Storage In Aquifer", key="Water_Storage_In_Aquifer",
                            on_change=Evaluation_Items_editor_change,
                            args=("Water_Storage_In_Aquifer", "Water_Storage_In_Aquifer"),
                            value=Evaluation_Items["Water_Storage_In_Aquifer"])
                st.checkbox("Depth Of Surface Water", key="Depth_Of_Surface_Water",
                            on_change=Evaluation_Items_editor_change,
                            args=("Depth_Of_Surface_Water", "Depth_Of_Surface_Water"),
                            value=Evaluation_Items["Depth_Of_Surface_Water"])
                st.checkbox("Groundwater Recharge Rate", key="Groundwater_Recharge_Rate",
                            on_change=Evaluation_Items_editor_change,
                            args=("Groundwater_Recharge_Rate", "Groundwater_Recharge_Rate"),
                            value=Evaluation_Items["Groundwater_Recharge_Rate"])

                st.write('##### :blue[Human Activity]')
                # st.subheader(":blue[]", divider=True)

                st.write('###### :blue[---urban---]')
                st.checkbox("Urban Anthropogenic Heat Flux", key="Urban_Anthropogenic_Heat_Flux",
                            on_change=Evaluation_Items_editor_change,
                            args=("Urban_Anthropogenic_Heat_Flux", "Urban_Anthropogenic_Heat_Flux"),
                            value=Evaluation_Items["Urban_Anthropogenic_Heat_Flux"])
                st.checkbox("Urban Albedo", key="Urban_Albedo", on_change=Evaluation_Items_editor_change,
                            args=("Urban_Albedo", "Urban_Albedo"), value=Evaluation_Items["Urban_Albedo"])
                st.checkbox("Urban Surface Temperature", key="Urban_Surface_Temperature",
                            on_change=Evaluation_Items_editor_change,
                            args=("Urban_Surface_Temperature", "Urban_Surface_Temperature"),
                            value=Evaluation_Items["Urban_Surface_Temperature"])
                st.checkbox("Urban Air Temperature Max", key="Urban_Air_Temperature_Max",
                            on_change=Evaluation_Items_editor_change,
                            args=("Urban_Air_Temperature_Max", "Urban_Air_Temperature_Max"),
                            value=Evaluation_Items["Urban_Air_Temperature_Max"])
                st.checkbox("Urban Air Temperature Min", key="Urban_Air_Temperature_Min",
                            on_change=Evaluation_Items_editor_change,
                            args=("Urban_Air_Temperature_Min", "Urban_Air_Temperature_Min"),
                            value=Evaluation_Items["Urban_Air_Temperature_Min"])
                st.checkbox("Urban Latent Heat Flux", key="Urban_Latent_Heat_Flux",
                            on_change=Evaluation_Items_editor_change,
                            args=("Urban_Latent_Heat_Flux", "Urban_Latent_Heat_Flux"),
                            value=Evaluation_Items["Urban_Latent_Heat_Flux"])
                st.write('###### :blue[---Crop---]')
                st.checkbox("Crop Yield Rice", key="Crop_Yield_Rice", on_change=Evaluation_Items_editor_change,
                            args=("Crop_Yield_Rice", "Crop_Yield_Rice"), value=Evaluation_Items["Crop_Yield_Rice"])
                st.checkbox("Crop Yield Corn", key="Crop_Yield_Corn", on_change=Evaluation_Items_editor_change,
                            args=("Crop_Yield_Corn", "Crop_Yield_Corn"), value=Evaluation_Items["Crop_Yield_Corn"])
                st.checkbox("Crop Yield Wheat", key="Crop_Yield_Wheat", on_change=Evaluation_Items_editor_change,
                            args=("Crop_Yield_Wheat", "Crop_Yield_Wheat"), value=Evaluation_Items["Crop_Yield_Wheat"])
                st.checkbox("Crop Yield Maize", key="Crop_Yield_Maize", on_change=Evaluation_Items_editor_change,
                            args=("Crop_Yield_Maize", "Crop_Yield_Maize"), value=Evaluation_Items["Crop_Yield_Maize"])

                st.checkbox("Crop Yield Soybean", key="Crop_Yield_Soybean", on_change=Evaluation_Items_editor_change,
                            args=("Crop_Yield_Soybean", "Crop_Yield_Soybean"),
                            value=Evaluation_Items["Crop_Yield_Soybean"])
                st.checkbox("Crop Heading DOY Corn", key="Crop_Heading_DOY_Corn",
                            on_change=Evaluation_Items_editor_change,
                            args=("Crop_Heading_DOY_Corn", "Crop_Heading_DOY_Corn"),
                            value=Evaluation_Items["Crop_Heading_DOY_Corn"])
                st.checkbox("Crop Heading DOY Wheat", key="Crop_Heading_DOY_Wheat",
                            on_change=Evaluation_Items_editor_change,
                            args=("Crop_Heading_DOY_Wheat", "Crop_Heading_DOY_Wheat"),
                            value=Evaluation_Items["Crop_Heading_DOY_Wheat"])
                st.checkbox("Crop Maturity DOY Corn", key="Crop_Maturity_DOY_Corn",
                            on_change=Evaluation_Items_editor_change,
                            args=("Crop_Maturity_DOY_Corn", "Crop_Maturity_DOY_Corn"),
                            value=Evaluation_Items["Crop_Maturity_DOY_Corn"])
                st.checkbox("Crop Maturity DOY Wheat", key="Crop_Maturity_DOY_Wheat",
                            on_change=Evaluation_Items_editor_change,
                            args=("Crop_Maturity_DOY_Wheat", "Crop_Maturity_DOY_Wheat"),
                            value=Evaluation_Items["Crop_Maturity_DOY_Wheat"])
                st.checkbox("Crop V3 DOY Corn", key="Crop_V3_DOY_Corn", on_change=Evaluation_Items_editor_change,
                            args=("Crop_V3_DOY_Corn", "Crop_V3_DOY_Corn"), value=Evaluation_Items["Crop_V3_DOY_Corn"])
                st.checkbox("Crop Emergence DOY Wheat", key="Crop_Emergence_DOY_Wheat",
                            on_change=Evaluation_Items_editor_change,
                            args=("Crop_Emergence_DOY_Wheat", "Crop_Emergence_DOY_Wheat"),
                            value=Evaluation_Items["Crop_Emergence_DOY_Wheat"])
                st.checkbox("Total Irrigation Amount", key="Total_Irrigation_Amount",
                            on_change=Evaluation_Items_editor_change,
                            args=("Total_Irrigation_Amount", "Total_Irrigation_Amount"),
                            value=Evaluation_Items["Total_Irrigation_Amount"])
                st.write('###### :blue[---Dam---]')
                st.checkbox("Dam Inflow", key="Dam_Inflow", on_change=Evaluation_Items_editor_change,
                            args=("Dam_Inflow", "Dam_Inflow"),
                            value=Evaluation_Items["Dam_Inflow"])
                st.checkbox("Dam Outflow", key="Dam_Outflow", on_change=Evaluation_Items_editor_change,
                            args=("Dam_Outflow", "Dam_Outflow"), value=Evaluation_Items["Dam_Outflow"])
                st.checkbox("Dam Water Storage", key="Dam_Water_Storage", on_change=Evaluation_Items_editor_change,
                            args=("Dam_Water_Storage", "Dam_Water_Storage"),
                            value=Evaluation_Items["Dam_Water_Storage"])

                st.checkbox("Dam Water Elevation", key="Dam_Water_Elevation", on_change=Evaluation_Items_editor_change,
                            args=("Dam_Water_Elevation", "Dam_Water_Elevation"),
                            value=Evaluation_Items["Dam_Water_Elevation"])
                st.write('###### :blue[---Lake---]')
                st.checkbox("Lake Temperature", key="Lake_Temperature", on_change=Evaluation_Items_editor_change,
                            args=("Lake_Temperature", "Lake_Temperature"), value=Evaluation_Items["Lake_Temperature"])
                st.checkbox("Lake Ice Fraction Cover", key="Lake_Ice_Fraction_Cover",
                            on_change=Evaluation_Items_editor_change,
                            args=("Lake_Ice_Fraction_Cover", "Lake_Ice_Fraction_Cover"),
                            value=Evaluation_Items["Lake_Ice_Fraction_Cover"])
                st.checkbox("Lake Water Level", key="Lake_Water_Level", on_change=Evaluation_Items_editor_change,
                            args=("Lake_Water_Level", "Lake_Water_Level"), value=Evaluation_Items["Lake_Water_Level"])
                st.checkbox("Lake Water Area", key="Lake_Water_Area", on_change=Evaluation_Items_editor_change,
                            args=("Lake_Water_Area", "Lake_Water_Area"), value=Evaluation_Items["Lake_Water_Area"])
                st.checkbox("Lake Water Volume", key="Lake_Water_Volume", on_change=Evaluation_Items_editor_change,
                            args=("Lake_Water_Volume", "Lake_Water_Volume"),
                            value=Evaluation_Items["Lake_Water_Volume"])
                st.write('###### :blue[---River---]')
                st.checkbox("Streamflow", key="Streamflow", on_change=Evaluation_Items_editor_change,
                            args=("Streamflow", "Streamflow"),
                            value=Evaluation_Items["Streamflow"])
                st.checkbox("Inundation Fraction", key="Inundation_Fraction", on_change=Evaluation_Items_editor_change,
                            args=("Inundation_Fraction", "Inundation_Fraction"),
                            value=Evaluation_Items["Inundation_Fraction"])
                st.checkbox("Inundation Area", key="Inundation_Area", on_change=Evaluation_Items_editor_change,
                            args=("Inundation_Area", "Inundation_Area"), value=Evaluation_Items["Inundation_Area"])
                st.checkbox("River Water Level", key="River_Water_Level", on_change=Evaluation_Items_editor_change,
                            args=("River_Water_Level", "River_Water_Level"),
                            value=Evaluation_Items["River_Water_Level"])

            return [item for item, value in Evaluation_Items.items() if value]

        def get_info(col, ilist):
            if isinstance(ilist, str): ilist = [ilist]
            case_item = {}
            for item in ilist:
                case_item[item] = False

            col.write('')
            with col.popover("Variables Infos", use_container_width=True):
                st.subheader(f"Showing Infos", divider=True)
                for item in ilist:
                    case_item[item] = st.checkbox(item, key=f"{item}__sim_Infos",
                                                  value=case_item[item])
                return [item for item, value in case_item.items() if value]

        if st.session_state.add_mod:
            self.__step3_add_mode()
        else:
            newlib['general'] = {}
            if newlib['Mod'] and not st.session_state.add_mod:
                Mod_nml = self.nl.read_namelist(os.path.join(self.Mod_variables_defination, f"{newlib['Mod']}.nml"))
                info_list = ['sub_dir', 'varname', 'varunit', 'prefix', 'suffix']
                st.divider()
                newlib['general']['model_namelist'] = self.sim_sources['def_Mod'][newlib['Mod']]
                col1, col2, col3 = st.columns(3)
                newlib['general']['timezone'] = col1.number_input(f"Set Time zone: ",
                                                                  value=0.0,
                                                                  key=f"new_simlib_timezone",
                                                                  on_change=sim_main_change,
                                                                  args=(f"timezone", 'new_simlib_timezone'),
                                                                  min_value=-12.0,
                                                                  max_value=12.0)
                newlib['general']['data_type'] = col2.selectbox(f'Set Data type: ',
                                                                options=('stn', 'Grid'),
                                                                index=1,
                                                                key=f"new_simlib_data_type",
                                                                on_change=sim_main_change,
                                                                args=(f"data_type", 'new_simlib_data_type'),
                                                                placeholder=f"Set your Simulation Data type...")
                newlib['general']['data_groupby'] = col3.selectbox(f'Set Data groupby: ',
                                                                   options=('hour', 'day', 'month', 'year', 'single'),
                                                                   index=4,
                                                                   key=f"new_simlib_data_groupby",
                                                                   on_change=sim_main_change,
                                                                   args=(f"data_groupby", 'new_simlib_data_groupby'),
                                                                   placeholder=f"Set your Simulation Data groupby...")
                newlib['general']['tim_res'] = col1.selectbox(f'Set Time Resolution: ',
                                                              options=('hour', 'day', 'month', 'year'),
                                                              index=0,
                                                              key=f"new_simlib_tim_res",
                                                              on_change=sim_main_change,
                                                              args=(f"tim_res", 'new_simlib_tim_res'),
                                                              placeholder=f"Set your Simulation Time Resolution ...")
                if newlib['general']['data_type'] == 'Grid':
                    newlib['general']['grid_res'] = col2.number_input(f"Set Geo Resolution: ",
                                                                      value=0.5,
                                                                      min_value=0.0,
                                                                      key=f"new_simlib_grid_res",
                                                                      on_change=sim_main_change,
                                                                      args=(f"grid_res", 'new_simlib_grid_res'),
                                                                      placeholder="Set your Simulation Geo Resolution...")

                    newlib['general']['syear'] = col1.number_input(f"Set Start year: ",
                                                                   value=2000,
                                                                   format='%04d', step=int(1),
                                                                   key=f"new_simlib_syear",
                                                                   on_change=sim_main_change,
                                                                   args=(f"syear", 'new_simlib_syear'),
                                                                   placeholder="Set your Simulation Start year...")
                    newlib['general']['eyear'] = col2.number_input(f"Set End year: ",
                                                                   value=2001,
                                                                   format='%04d', step=int(1),
                                                                   key=f"new_simlib_eyear",
                                                                   on_change=sim_main_change,
                                                                   args=(f"eyear", 'new_simlib_eyear'),
                                                                   placeholder="Set your Simulation End year...")
                    ffix = st.toggle(f'Simulation {newlib["Sim_casename"]} share the same prefix and suffix?', value=True)
                    col1, col2, col3 = st.columns(3)
                    if ffix:
                        info_list = ['sub_dir', 'varname', 'varunit']
                        newlib['general']['prefix'] = col1.text_input(f'Set File prefix: ',
                                                                      value='',
                                                                      key=f"new_simlib_prefix",
                                                                      on_change=sim_main_change,
                                                                      args=(f"prefix", 'new_simlib_prefix'),
                                                                      placeholder=f"Set your Simulation prefix...")
                        newlib['general']['suffix'] = col2.text_input(f'Set File suffix: ',
                                                                      value='',
                                                                      key=f"new_simlib_suffix",
                                                                      on_change=sim_main_change,
                                                                      args=(f"suffix", 'new_simlib_suffix'),
                                                                      placeholder=f"Set your Simulation suffix...")

                    newlib['general'][f"fulllist"] = ''
                else:
                    info_list = ['sub_dir', 'varname', 'varunit']
                    if 'fulllist' not in newlib['general']: newlib['general']['fulllist'] = None
                    if not newlib['general'][f"fulllist"]: newlib['general'][f"fulllist"] = None
                    newlib['general'][f"fulllist"] = self.path_finder.get_file(newlib['general'][f"fulllist"],
                                                                               f"new_simlib_fulllist",
                                                                               'csv', [None, None])
                    st.code(f"Set Fulllist File: {newlib['general'][f'fulllist']}", language='shell', wrap_lines=True)

                    newlib['general']['grid_res'] = ''
                    newlib['general']['syear'] = ''
                    newlib['general']['eyear'] = ''
                    newlib['general']['suffix'] = ''
                    newlib['general']['prefix'] = ''

                if 'root_dir' not in newlib['general']: newlib['general']['root_dir'] = './data/'
                if not newlib['general']['root_dir']: newlib['general']['root_dir'] = './data/'
                try:
                    newlib['general']['root_dir'] = self.path_finder.find_path(newlib['general']['root_dir'],
                                                                               f"new_simlib_{newlib['Sim_casename']}_root_dir",
                                                                               [None, None])
                    st.code(f"Set Data Dictionary: {newlib['general']['root_dir']}", language='shell', wrap_lines=True)
                except PermissionError as e:
                    if e:
                        newlib['general']['root_dir'] = '/'

                st.divider()
                st.write(f'###### :orange[:point_down: If some item is differ to {newlib["Mod"]} variables, press to change]')
                col1, col2 = st.columns((2, 1.2))

                newlib['variables'] = get_var(col1, 'case_items')
                newlib['info_list'] = get_info(col2, info_list)
                if newlib['variables'] and newlib['info_list']:
                    info_lists = {'sub_dir': {'title': 'Set Data Sub-Dictionary', 'value': ''},
                                  'varname': {'title': 'Set varname', 'value': ''},
                                  'varunit': {'title': 'Set varunit', 'value': ''},
                                  'prefix': {'title': 'Set prefix', 'value': ''},
                                  'suffix': {'title': 'Set suffix', 'value': ''},
                                  'syear': {'title': 'Set syear', 'value': 2000},
                                  'eyear': {'title': 'Set eyear', 'value': 2001}}

                    for var in self.evaluation_items:
                        if var in newlib['variables']:
                            newlib[var] = {}
                            for info in newlib['info_list']:
                                newlib[var][info] = info_lists[info]['value']
                                if var in Mod_nml and info in Mod_nml[var]:
                                    newlib[var][info] = Mod_nml[var][info]
                        if var in newlib and var not in newlib['variables']:
                            del newlib[var]

                    import itertools
                    for variable in newlib['variables']:
                        with st.container(height=None, border=True):
                            st.write(f"##### :blue[{variable.replace('_', ' ')}]")
                            if variable not in Mod_nml:
                                st.info(f"###### :orange[{variable.replace('_', ' ')} not in {newlib['Mod']}, remeber to set varname and units]")
                            cols = itertools.cycle(st.columns(3))
                            for info in newlib['info_list']:
                                if info == 'sub_dir':
                                    with st.container():
                                        if 'sub_dir' not in newlib[variable]: newlib[variable][info] = newlib[variable][info]
                                        if not newlib[variable][info]: newlib[variable][info] = ''
                                        newlib[variable][info] = self.path_finder.find_subdirectories(newlib[variable][info],
                                                                                                      f"new_simlib_{newlib['Sim_casename']}_root_dir",
                                                                                                      f"{variable}_{info}_sim",
                                                                                                      [None, None])
                                        st.code(f"Sub-Dir: {newlib[variable][info]}", language='shell', wrap_lines=True)
                                else:
                                    col = next(cols)
                                    newlib[variable][info] = col.text_input(info_lists[info]['title'],
                                                                            value=newlib[variable][info],
                                                                            key=f"{variable}_{info}_sim",
                                                                            placeholder=f"Set your Simulation Var...")

            disable = False
            check = self.__step3_check_newcase_namelist(newlib)
            if not check:
                disable = True

            def define_add():
                st.session_state.step3_add_nml = True

            col1, col2, col3 = st.columns(3)
            if col1.button('Make namelist', help='Yes, this is the one.', disabled=disable, on_click=define_add):
                self.__step3_make_case_namelist(sim_save_path, newlib)
                st.success("😉 Make file successfully!!! \n Please press to Next step")

        st.divider()

        def define_back(make_contain, warn):
            if not disable:
                if not st.session_state.step3_add_nml:
                    with make_contain:
                        self.__step3_make_case_namelist(sim_save_path, newlib)
                    make_contain.success("😉 Make file successfully!!! \n Please press to Next step")
                st.session_state.step3_add_nml = False
            else:
                make_contain.warning('Making file failed or No file is processing!')
                time.sleep(0.8)
            st.session_state.step3_make_newnamelist = False
            del_vars = ['case_items']
            for del_var in del_vars:
                del st.session_state[del_var]

        make_contain = st.container()
        st.button('Finish and back to select page', on_click=define_back, args=(make_contain, 'make_contain'),
                  help='Finish add and go back to select page')

    def __check_path(self):
        path = os.path.join(st.session_state.openbench_path, 'nml', 'user')
        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

        st.markdown(f"""
        <div style="font-size:22px; font-weight:bold; color:#68838B; border-bottom:3px solid #68838B; padding: 5px;">
            Set Simulation file 👇
        </div>""", unsafe_allow_html=True)
        st.write(' ')
        file = st.radio("Set Simulation file 👇",
                        ["user"] + [f"user/{folder}" for folder in folders] + ['New folder'],
                        key="sim_savefile", label_visibility='collapsed',
                        horizontal=True)

        if file == 'New folder':
            col1, col2 = st.columns(2)
            col1.write('##### :green[Simulation new file name]')
            name = col1.text_input(f'Simulation file name: ', value='',
                                   key=f"Sim_filename",
                                   placeholder=f"Set your Simulation file...")
            if name:
                file = f"user/{name}"
                os.makedirs(os.path.join(st.session_state.namelist_path, file), exist_ok=True)  # 创建目录（如果不存在）
                return os.path.join(st.session_state.openbench_path, 'nml', file)
        else:
            return os.path.join(st.session_state.openbench_path, 'nml', file)

    def __step3_add_mode(self):

        def get_var(col, name):
            if name not in st.session_state:
                st.session_state[name] = self.initial.evaluation_items()
            Evaluation_Items = st.session_state[name]
            col.write('')
            with col.popover("Variables items", use_container_width=True):
                def Evaluation_Items_editor_change(key, editor_key):
                    Evaluation_Items[key] = st.session_state[key]

                st.subheader("Mod Variables ....", divider=True)
                st.write('##### :blue[Ecosystem and Carbon Cycle]')
                # st.subheader("", divider=True, )
                st.checkbox("Gross Primary Productivity", key="Gross_Primary_Productivity",
                            on_change=Evaluation_Items_editor_change,
                            args=("Gross_Primary_Productivity", "Gross_Primary_Productivity"),
                            value=Evaluation_Items["Gross_Primary_Productivity"])
                st.checkbox("Ecosystem Respiration", key="Ecosystem_Respiration",
                            on_change=Evaluation_Items_editor_change,
                            args=("Ecosystem_Respiration", "Ecosystem_Respiration"),
                            value=Evaluation_Items["Ecosystem_Respiration"])
                st.checkbox("Net Ecosystem Exchange", key="Net_Ecosystem_Exchange",
                            on_change=Evaluation_Items_editor_change,
                            args=("Net_Ecosystem_Exchange", "Net_Ecosystem_Exchange"),
                            value=Evaluation_Items["Net_Ecosystem_Exchange"])
                st.checkbox("Leaf Area Index", key="Leaf_Area_Index", on_change=Evaluation_Items_editor_change,
                            args=("Leaf_Area_Index", "Leaf_Area_Index"), value=Evaluation_Items["Leaf_Area_Index"])
                st.checkbox("Biomass", key="Biomass", on_change=Evaluation_Items_editor_change,
                            args=("Biomass", "Biomass"),
                            value=Evaluation_Items["Biomass"])
                st.checkbox("Burned Area", key="Burned_Area", on_change=Evaluation_Items_editor_change,
                            args=("Burned_Area", "Burned_Area"), value=Evaluation_Items["Burned_Area"])
                st.checkbox("Soil Carbon", key="Soil_Carbon", on_change=Evaluation_Items_editor_change,
                            args=("Soil_Carbon", "Soil_Carbon"), value=Evaluation_Items["Soil_Carbon"])
                st.checkbox("Nitrogen Fixation", key="Nitrogen_Fixation", on_change=Evaluation_Items_editor_change,
                            args=("Nitrogen_Fixation", "Nitrogen_Fixation"),
                            value=Evaluation_Items["Nitrogen_Fixation"])
                st.checkbox("Methane", key="Methane", on_change=Evaluation_Items_editor_change,
                            args=("Methane", "Methane"),
                            value=Evaluation_Items["Methane"])
                st.checkbox("Veg Cover In Fraction", key="Veg_Cover_In_Fraction",
                            on_change=Evaluation_Items_editor_change,
                            args=("Veg_Cover_In_Fraction", "Veg_Cover_In_Fraction"),
                            value=Evaluation_Items["Veg_Cover_In_Fraction"])
                st.checkbox("Leaf Greenness", key="Leaf_Greenness", on_change=Evaluation_Items_editor_change,
                            args=("Leaf_Greenness", "Leaf_Greenness"), value=Evaluation_Items["Leaf_Greenness"])

                st.write('##### :blue[Radiation and Energy Cycle]')
                st.checkbox("Net Radiation", key="Net_Radiation", on_change=Evaluation_Items_editor_change,
                            args=("Net_Radiation", "Net_Radiation"), value=Evaluation_Items["Net_Radiation"])
                st.checkbox("Latent Heat", key="Latent_Heat", on_change=Evaluation_Items_editor_change,
                            args=("Latent_Heat", "Latent_Heat"), value=Evaluation_Items["Latent_Heat"])
                st.checkbox("Sensible Heat", key="Sensible_Heat", on_change=Evaluation_Items_editor_change,
                            args=("Sensible_Heat", "Sensible_Heat"), value=Evaluation_Items["Sensible_Heat"])
                st.checkbox("Ground Heat", key="Ground_Heat", on_change=Evaluation_Items_editor_change,
                            args=("Ground_Heat", "Ground_Heat"), value=Evaluation_Items["Ground_Heat"])
                st.checkbox("Albedo", key="Albedo", on_change=Evaluation_Items_editor_change, args=("Albedo", "Albedo"),
                            value=Evaluation_Items["Albedo"])
                st.checkbox("Surface Upward SW Radiation", key="Surface_Upward_SW_Radiation",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Upward_SW_Radiation", "Surface_Upward_SW_Radiation"),
                            value=Evaluation_Items["Surface_Upward_SW_Radiation"])
                st.checkbox("Surface Upward LW Radiation", key="Surface_Upward_LW_Radiation",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Upward_LW_Radiation", "Surface_Upward_LW_Radiation"),
                            value=Evaluation_Items["Surface_Upward_LW_Radiation"])
                st.checkbox("Surface Net SW Radiation", key="Surface_Net_SW_Radiation",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Net_SW_Radiation", "Surface_Net_SW_Radiation"),
                            value=Evaluation_Items["Surface_Net_SW_Radiation"])
                st.checkbox("Surface Net LW Radiation", key="Surface_Net_LW_Radiation",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Net_LW_Radiation", "Surface_Net_LW_Radiation"),
                            value=Evaluation_Items["Surface_Net_LW_Radiation"])
                st.checkbox("Surface Soil Temperature", key="Surface_Soil_Temperature",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Soil_Temperature", "Surface_Soil_Temperature"),
                            value=Evaluation_Items["Surface_Soil_Temperature"])
                st.checkbox("Root Zone Soil Temperature", key="Root_Zone_Soil_Temperature",
                            on_change=Evaluation_Items_editor_change,
                            args=("Root_Zone_Soil_Temperature", "Root_Zone_Soil_Temperature"),
                            value=Evaluation_Items["Root_Zone_Soil_Temperature"])

                st.write('##### :blue[Forcings]')
                # st.subheader(":blue[]", divider=True)
                st.checkbox("Diurnal Temperature Range", key="Diurnal_Temperature_Range",
                            on_change=Evaluation_Items_editor_change,
                            args=("Diurnal_Temperature_Range", "Diurnal_Temperature_Range"),
                            value=Evaluation_Items["Diurnal_Temperature_Range"])
                st.checkbox("Diurnal Max Temperature", key="Diurnal_Max_Temperature",
                            on_change=Evaluation_Items_editor_change,
                            args=("Diurnal_Max_Temperature", "Diurnal_Max_Temperature"),
                            value=Evaluation_Items["Diurnal_Max_Temperature"])
                st.checkbox("Diurnal Min Temperature", key="Diurnal_Min_Temperature",
                            on_change=Evaluation_Items_editor_change,
                            args=("Diurnal_Min_Temperature", "Diurnal_Min_Temperature"),
                            value=Evaluation_Items["Diurnal_Min_Temperature"])
                st.checkbox("Surface Downward SW Radiation", key="Surface_Downward_SW_Radiation",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Downward_SW_Radiation", "Surface_Downward_SW_Radiation"),
                            value=Evaluation_Items["Surface_Downward_SW_Radiation"])
                st.checkbox("Surface Downward LW Radiation", key="Surface_Downward_LW_Radiation",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Downward_LW_Radiation", "Surface_Downward_LW_Radiation"),
                            value=Evaluation_Items["Surface_Downward_LW_Radiation"])
                st.checkbox("Surface Relative Humidity", key="Surface_Relative_Humidity",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Relative_Humidity", "Surface_Relative_Humidity"),
                            value=Evaluation_Items["Surface_Relative_Humidity"])
                st.checkbox("Surface Specific Humidity", key="Surface_Specific_Humidity",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Specific_Humidity", "Surface_Specific_Humidity"),
                            value=Evaluation_Items["Surface_Specific_Humidity"])
                st.checkbox("Precipitation", key="Precipitation", on_change=Evaluation_Items_editor_change,
                            args=("Precipitation", "Precipitation"), value=Evaluation_Items["Precipitation"])
                st.checkbox("Surface Air Temperature", key="Surface_Air_Temperature",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Air_Temperature", "Surface_Air_Temperature"),
                            value=Evaluation_Items["Surface_Air_Temperature"])

                st.write('##### :blue[Hydrology Cycle]')
                # st.subheader(":blue[]", divider=True)

                st.checkbox("Evapotranspiration", key="Evapotranspiration", on_change=Evaluation_Items_editor_change,
                            args=("Evapotranspiration", "Evapotranspiration"),
                            value=Evaluation_Items["Evapotranspiration"])
                st.checkbox("Canopy Transpiration", key="Canopy_Transpiration",
                            on_change=Evaluation_Items_editor_change,
                            args=("Canopy_Transpiration", "Canopy_Transpiration"),
                            value=Evaluation_Items["Canopy_Transpiration"])
                st.checkbox("Canopy Interception", key="Canopy_Interception",
                            on_change=Evaluation_Items_editor_change,
                            args=("Canopy_Interception", "Canopy_Interception"),
                            value=Evaluation_Items["Canopy_Interception"])
                st.checkbox("Ground Evaporation", key="Ground_Evaporation", on_change=Evaluation_Items_editor_change,
                            args=("Ground_Evaporation", "Ground_Evaporation"),
                            value=Evaluation_Items["Ground_Evaporation"])
                st.checkbox("Water Evaporation", key="Water_Evaporation", on_change=Evaluation_Items_editor_change,
                            args=("Water_Evaporation", "Water_Evaporation"),
                            value=Evaluation_Items["Water_Evaporation"])
                st.checkbox("Soil Evaporation", key="Soil_Evaporation", on_change=Evaluation_Items_editor_change,
                            args=("Soil_Evaporation", "Soil_Evaporation"), value=Evaluation_Items["Soil_Evaporation"])
                st.checkbox("Total Runoff", key="Total_Runoff", on_change=Evaluation_Items_editor_change,
                            args=("Total_Runoff", "Total_Runoff"), value=Evaluation_Items["Total_Runoff"])
                st.checkbox("Terrestrial Water Storage Change", key="Terrestrial_Water_Storage_Change",
                            on_change=Evaluation_Items_editor_change,
                            args=("Terrestrial_Water_Storage_Change", "Terrestrial_Water_Storage_Change"),
                            value=Evaluation_Items["Terrestrial_Water_Storage_Change"])
                st.checkbox("Snow Water Equivalent", key="Snow_Water_Equivalent",
                            on_change=Evaluation_Items_editor_change,
                            args=("Snow_Water_Equivalent", "Snow_Water_Equivalent"),
                            value=Evaluation_Items["Snow_Water_Equivalent"])

                st.checkbox("Surface Snow Cover In Fraction", key="Surface_Snow_Cover_In_Fraction",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Snow_Cover_In_Fraction", "Surface_Snow_Cover_In_Fraction"),
                            value=Evaluation_Items["Surface_Snow_Cover_In_Fraction"])
                st.checkbox("Snow Depth", key="Snow_Depth", on_change=Evaluation_Items_editor_change,
                            args=("Snow_Depth", "Snow_Depth"), value=Evaluation_Items["Snow_Depth"])
                st.checkbox("Permafrost", key="Permafrost", on_change=Evaluation_Items_editor_change,
                            args=("Permafrost", "Permafrost"), value=Evaluation_Items["Permafrost"])
                st.checkbox("Surface Soil Moisture", key="Surface_Soil_Moisture",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Soil_Moisture", "Surface_Soil_Moisture"),
                            value=Evaluation_Items["Surface_Soil_Moisture"])
                st.checkbox("Root Zone Soil Moisture", key="Root_Zone_Soil_Moisture",
                            on_change=Evaluation_Items_editor_change,
                            args=("Root_Zone_Soil_Moisture", "Root_Zone_Soil_Moisture"),
                            value=Evaluation_Items["Root_Zone_Soil_Moisture"])
                st.checkbox("Water Table Depth", key="Water_Table_Depth", on_change=Evaluation_Items_editor_change,
                            args=("Water_Table_Depth", "Water_Table_Depth"),
                            value=Evaluation_Items["Water_Table_Depth"])
                st.checkbox("Water Storage In Aquifer", key="Water_Storage_In_Aquifer",
                            on_change=Evaluation_Items_editor_change,
                            args=("Water_Storage_In_Aquifer", "Water_Storage_In_Aquifer"),
                            value=Evaluation_Items["Water_Storage_In_Aquifer"])
                st.checkbox("Depth Of Surface Water", key="Depth_Of_Surface_Water",
                            on_change=Evaluation_Items_editor_change,
                            args=("Depth_Of_Surface_Water", "Depth_Of_Surface_Water"),
                            value=Evaluation_Items["Depth_Of_Surface_Water"])
                st.checkbox("Groundwater Recharge Rate", key="Groundwater_Recharge_Rate",
                            on_change=Evaluation_Items_editor_change,
                            args=("Groundwater_Recharge_Rate", "Groundwater_Recharge_Rate"),
                            value=Evaluation_Items["Groundwater_Recharge_Rate"])

                st.write('##### :blue[Human Activity]')
                # st.subheader(":blue[]", divider=True)

                st.write('###### :blue[---urban---]')
                st.checkbox("Urban Anthropogenic Heat Flux", key="Urban_Anthropogenic_Heat_Flux",
                            on_change=Evaluation_Items_editor_change,
                            args=("Urban_Anthropogenic_Heat_Flux", "Urban_Anthropogenic_Heat_Flux"),
                            value=Evaluation_Items["Urban_Anthropogenic_Heat_Flux"])
                st.checkbox("Urban Albedo", key="Urban_Albedo", on_change=Evaluation_Items_editor_change,
                            args=("Urban_Albedo", "Urban_Albedo"), value=Evaluation_Items["Urban_Albedo"])
                st.checkbox("Urban Surface Temperature", key="Urban_Surface_Temperature",
                            on_change=Evaluation_Items_editor_change,
                            args=("Urban_Surface_Temperature", "Urban_Surface_Temperature"),
                            value=Evaluation_Items["Urban_Surface_Temperature"])
                st.checkbox("Urban Air Temperature Max", key="Urban_Air_Temperature_Max",
                            on_change=Evaluation_Items_editor_change,
                            args=("Urban_Air_Temperature_Max", "Urban_Air_Temperature_Max"),
                            value=Evaluation_Items["Urban_Air_Temperature_Max"])
                st.checkbox("Urban Air Temperature Min", key="Urban_Air_Temperature_Min",
                            on_change=Evaluation_Items_editor_change,
                            args=("Urban_Air_Temperature_Min", "Urban_Air_Temperature_Min"),
                            value=Evaluation_Items["Urban_Air_Temperature_Min"])
                st.checkbox("Urban Latent Heat Flux", key="Urban_Latent_Heat_Flux",
                            on_change=Evaluation_Items_editor_change,
                            args=("Urban_Latent_Heat_Flux", "Urban_Latent_Heat_Flux"),
                            value=Evaluation_Items["Urban_Latent_Heat_Flux"])
                st.write('###### :blue[---Crop---]')
                st.checkbox("Crop Yield Rice", key="Crop_Yield_Rice", on_change=Evaluation_Items_editor_change,
                            args=("Crop_Yield_Rice", "Crop_Yield_Rice"), value=Evaluation_Items["Crop_Yield_Rice"])
                st.checkbox("Crop Yield Corn", key="Crop_Yield_Corn", on_change=Evaluation_Items_editor_change,
                            args=("Crop_Yield_Corn", "Crop_Yield_Corn"), value=Evaluation_Items["Crop_Yield_Corn"])
                st.checkbox("Crop Yield Wheat", key="Crop_Yield_Wheat", on_change=Evaluation_Items_editor_change,
                            args=("Crop_Yield_Wheat", "Crop_Yield_Wheat"), value=Evaluation_Items["Crop_Yield_Wheat"])
                st.checkbox("Crop Yield Maize", key="Crop_Yield_Maize", on_change=Evaluation_Items_editor_change,
                            args=("Crop_Yield_Maize", "Crop_Yield_Maize"), value=Evaluation_Items["Crop_Yield_Maize"])

                st.checkbox("Crop Yield Soybean", key="Crop_Yield_Soybean", on_change=Evaluation_Items_editor_change,
                            args=("Crop_Yield_Soybean", "Crop_Yield_Soybean"),
                            value=Evaluation_Items["Crop_Yield_Soybean"])
                st.checkbox("Crop Heading DOY Corn", key="Crop_Heading_DOY_Corn",
                            on_change=Evaluation_Items_editor_change,
                            args=("Crop_Heading_DOY_Corn", "Crop_Heading_DOY_Corn"),
                            value=Evaluation_Items["Crop_Heading_DOY_Corn"])
                st.checkbox("Crop Heading DOY Wheat", key="Crop_Heading_DOY_Wheat",
                            on_change=Evaluation_Items_editor_change,
                            args=("Crop_Heading_DOY_Wheat", "Crop_Heading_DOY_Wheat"),
                            value=Evaluation_Items["Crop_Heading_DOY_Wheat"])
                st.checkbox("Crop Maturity DOY Corn", key="Crop_Maturity_DOY_Corn",
                            on_change=Evaluation_Items_editor_change,
                            args=("Crop_Maturity_DOY_Corn", "Crop_Maturity_DOY_Corn"),
                            value=Evaluation_Items["Crop_Maturity_DOY_Corn"])
                st.checkbox("Crop Maturity DOY Wheat", key="Crop_Maturity_DOY_Wheat",
                            on_change=Evaluation_Items_editor_change,
                            args=("Crop_Maturity_DOY_Wheat", "Crop_Maturity_DOY_Wheat"),
                            value=Evaluation_Items["Crop_Maturity_DOY_Wheat"])
                st.checkbox("Crop V3 DOY Corn", key="Crop_V3_DOY_Corn", on_change=Evaluation_Items_editor_change,
                            args=("Crop_V3_DOY_Corn", "Crop_V3_DOY_Corn"), value=Evaluation_Items["Crop_V3_DOY_Corn"])
                st.checkbox("Crop Emergence DOY Wheat", key="Crop_Emergence_DOY_Wheat",
                            on_change=Evaluation_Items_editor_change,
                            args=("Crop_Emergence_DOY_Wheat", "Crop_Emergence_DOY_Wheat"),
                            value=Evaluation_Items["Crop_Emergence_DOY_Wheat"])
                st.checkbox("Total Irrigation Amount", key="Total_Irrigation_Amount",
                            on_change=Evaluation_Items_editor_change,
                            args=("Total_Irrigation_Amount", "Total_Irrigation_Amount"),
                            value=Evaluation_Items["Total_Irrigation_Amount"])
                st.write('###### :blue[---Dam---]')
                st.checkbox("Dam Inflow", key="Dam_Inflow", on_change=Evaluation_Items_editor_change,
                            args=("Dam_Inflow", "Dam_Inflow"),
                            value=Evaluation_Items["Dam_Inflow"])
                st.checkbox("Dam Outflow", key="Dam_Outflow", on_change=Evaluation_Items_editor_change,
                            args=("Dam_Outflow", "Dam_Outflow"), value=Evaluation_Items["Dam_Outflow"])
                st.checkbox("Dam Water Storage", key="Dam_Water_Storage", on_change=Evaluation_Items_editor_change,
                            args=("Dam_Water_Storage", "Dam_Water_Storage"),
                            value=Evaluation_Items["Dam_Water_Storage"])

                st.checkbox("Dam Water Elevation", key="Dam_Water_Elevation", on_change=Evaluation_Items_editor_change,
                            args=("Dam_Water_Elevation", "Dam_Water_Elevation"),
                            value=Evaluation_Items["Dam_Water_Elevation"])
                st.write('###### :blue[---Lake---]')
                st.checkbox("Lake Temperature", key="Lake_Temperature", on_change=Evaluation_Items_editor_change,
                            args=("Lake_Temperature", "Lake_Temperature"), value=Evaluation_Items["Lake_Temperature"])
                st.checkbox("Lake Ice Fraction Cover", key="Lake_Ice_Fraction_Cover",
                            on_change=Evaluation_Items_editor_change,
                            args=("Lake_Ice_Fraction_Cover", "Lake_Ice_Fraction_Cover"),
                            value=Evaluation_Items["Lake_Ice_Fraction_Cover"])
                st.checkbox("Lake Water Level", key="Lake_Water_Level", on_change=Evaluation_Items_editor_change,
                            args=("Lake_Water_Level", "Lake_Water_Level"), value=Evaluation_Items["Lake_Water_Level"])
                st.checkbox("Lake Water Area", key="Lake_Water_Area", on_change=Evaluation_Items_editor_change,
                            args=("Lake_Water_Area", "Lake_Water_Area"), value=Evaluation_Items["Lake_Water_Area"])
                st.checkbox("Lake Water Volume", key="Lake_Water_Volume", on_change=Evaluation_Items_editor_change,
                            args=("Lake_Water_Volume", "Lake_Water_Volume"),
                            value=Evaluation_Items["Lake_Water_Volume"])
                st.write('###### :blue[---River---]')
                st.checkbox("Streamflow", key="Streamflow", on_change=Evaluation_Items_editor_change,
                            args=("Streamflow", "Streamflow"),
                            value=Evaluation_Items["Streamflow"])
                st.checkbox("Inundation Fraction", key="Inundation_Fraction", on_change=Evaluation_Items_editor_change,
                            args=("Inundation_Fraction", "Inundation_Fraction"),
                            value=Evaluation_Items["Inundation_Fraction"])
                st.checkbox("Inundation Area", key="Inundation_Area", on_change=Evaluation_Items_editor_change,
                            args=("Inundation_Area", "Inundation_Area"), value=Evaluation_Items["Inundation_Area"])
                st.checkbox("River Water Level", key="River_Water_Level", on_change=Evaluation_Items_editor_change,
                            args=("River_Water_Level", "River_Water_Level"),
                            value=Evaluation_Items["River_Water_Level"])

            return [item for item, value in Evaluation_Items.items() if value]

        st.markdown(f"""
                    <div style="font-size:22px; font-weight:bold; color:#68838B; border-bottom:3px solid #68838B; padding: 5px;">
                        Add Mod variables defination
                    </div>""", unsafe_allow_html=True)
        st.write(' ')
        mod_lib = {}
        mod_lib['general'] = {}
        container = st.container(height=None, border=False)
        col1, col2, col3 = container.columns((3, 3.5, 2.5))
        col1.write('###### :blue[Model name]')
        col2.write('###### :blue[Select variables]')
        col3.write('###### :blue[Make Mod namelist]')

        col1, col2, col3 = container.columns((2.5, 4, 2.5))
        mod_lib['general']['model'] = col1.text_input(f'Model name: ', value='', label_visibility='collapsed',
                                                      placeholder=f"Set your Model name...")
        variables = get_var(col2, 'mode_item')
        for var in self.evaluation_items:
            if var in variables:
                mod_lib[var] = {}
            if var in mod_lib and var not in variables:
                del mod_lib[var]

        info_lists = {
            'varname': {'title': 'Set varname', 'value': ''},
            'varunit': {'title': 'Set varunit', 'value': ''}
        }
        with st.expander(' :orange[Add Mod variables defination]', expanded=True, icon=None):
            import itertools
            info_list = ['varname', 'varunit']
            tcols = itertools.cycle(st.columns(2))
            for variable in variables:
                tcol = next(tcols)
                tcol.write(f"##### :blue[{variable.replace('_', ' ')}]")
                cols = itertools.cycle(tcol.columns(2))
                for info in info_list:
                    col = next(cols)
                    mod_lib[variable][info] = col.text_input(info_lists[info]['title'],
                                                             value=info_lists[info]['value'],
                                                             key=f"{variable}_{info}_model",
                                                             placeholder=f"Set your Model Var...")
                tcol.divider()
        col3.write('''''')
        if col3.button('Make Mod namelist', help='Yes, this is the one.'):
            make_Mod = self.__step3_make_Mod_namelist(self.Mod_variables_defination, mod_lib, variables)
            if make_Mod:
                st.success("😉 Make file successfully!!! \n Please press to Next step")

    def __step3_make_Mod_namelist(self, path, mod_lib, variables):
        """
        Write a namelist from a text file.

        Args:
            file_path (str): Path to the text file.

        """
        if mod_lib["general"]["model"] == '':
            st.error('Please define your model first!')
            return False
        elif len(variables) == 0:
            st.error('Please choose variables first!')
            return False
        else:
            with st.spinner('Making namelist... Please wait.'):
                with open(path + f'/{mod_lib["general"]["model"]}.nml', 'w') as f:
                    lines = []
                    end_line = "/\n\n\n"

                    lines.append("&general\n")
                    lines.append(f"    model = {mod_lib['general']['model']}\n")
                    lines.append(end_line)

                    for variable in variables:
                        lines.append(f"&{variable}\n")
                        for info in ['varname', 'varunit']:
                            lines.append(f"    {info} = {mod_lib[variable][info]}\n")
                        lines.append(end_line)
                    for line in lines:
                        f.write(line)
                    time.sleep(2)

                model_path = os.path.join(path, f'{mod_lib["general"]["model"]}.nml')
                self.sim_sources['def_Mod'][mod_lib["general"]["model"]] = self.path_finder.check_rel_path(model_path)
                with open('./GUI/Namelist_lib/Simulation_lib.nml', 'w') as f1:
                    lines = []
                    end_line = "/\n\n\n"
                    lines.append("&general\n")
                    max_key_length = max(len(key) for key in self.sim_sources['general'].keys())
                    for key in list(self.sim_sources['general'].keys()):
                        value = self.sim_sources['general'][f'{key}']
                        if isinstance(value, str): value = [value]
                        lines.append(f"    {key:<{max_key_length}} = {', '.join(value)}\n")
                    lines.append(end_line)

                    lines.append("&def_nml\n")
                    max_key_length = max(len(key) for key in self.sim_sources['def_nml'].keys())
                    for key in list(self.sim_sources['def_nml'].keys()):
                        lines.append(f"    {key:<{max_key_length}} = {self.sim_sources['def_nml'][f'{key}']}\n")
                    lines.append(end_line)

                    lines.append("&def_Mod\n")
                    max_key_length = max(len(key) for key in self.sim_sources['def_Mod'].keys())
                    for key in list(self.sim_sources['def_Mod'].keys()):
                        lines.append(f"    {key:<{max_key_length}} = {self.sim_sources['def_Mod'][f'{key}']}\n")
                    lines.append(end_line)

                    for line in lines:
                        f1.write(line)
                time.sleep(2)
                return False

    def __step3_make_case_namelist(self, sim_save_path, newlib):
        """
        Write a namelist from a text file.

        Args:
            file_path (str): Path to the text file.

        """

        with (st.spinner('Making namelist... Please wait.')):
            with open(sim_save_path + f'/{newlib["Sim_casename"]}.nml', 'w') as f:
                lines = []
                end_line = "/\n\n\n"

                lines.append("&general\n")
                max_key_length = max(len(key) for key in newlib['general'].keys())
                for key in list(newlib['general'].keys()):
                    lines.append(f"    {key:<{max_key_length}} = {newlib['general'][f'{key}']}\n")
                lines.append(end_line)

                if newlib['variables']:
                    if isinstance(newlib['variables'], str): newlib['variables'] = [newlib['variables']]
                    for variable in newlib['variables']:
                        variable = variable.replace(' ', '_')
                        lines.append(f"&{variable}\n")
                        for info in newlib['info_list']:
                            if newlib[variable][info] is None:
                                lines.append(f"    {info} = \n")
                            else:
                                lines.append(f"    {info} = {newlib[variable][info]}\n")
                        lines.append(end_line)
                for line in lines:
                    f.write(line)
                time.sleep(2)

            if isinstance(self.sim_sources['general']['Case_lib'], str): self.sim_sources['general']['Case_lib'] = [
                self.sim_sources['general']['Case_lib']]
            if newlib["Sim_casename"] not in self.sim_sources['general']['Case_lib']:
                self.sim_sources['general']['Case_lib'].append(newlib["Sim_casename"])
            case_path = os.path.join(sim_save_path, f'{newlib["Sim_casename"]}.nml')
            self.sim_sources['def_nml'][newlib["Sim_casename"]] = self.path_finder.check_rel_path(case_path)

            with open('./GUI/Namelist_lib/Simulation_lib.nml', 'w') as f1:
                lines = []
                end_line = "/\n\n\n"
                lines.append("&general\n")
                max_key_length = max(len(key) for key in self.sim_sources['general'].keys())
                for key, value in self.sim_sources['general'].items():
                    if isinstance(value, str): value = [value]
                    lines.append(f"    {key:<{max_key_length}} = {', '.join(value)}\n")
                lines.append(end_line)

                lines.append("&def_nml\n")
                max_key_length = max(len(key) for key in self.sim_sources['def_nml'].keys())
                for key in list(self.sim_sources['def_nml'].keys()):
                    lines.append(f"    {key:<{max_key_length}} = {self.sim_sources['def_nml'][f'{key}']}\n")
                lines.append(end_line)

                lines.append("&def_Mod\n")
                max_key_length = max(len(key) for key in self.sim_sources['def_Mod'].keys())
                for key in list(self.sim_sources['def_Mod'].keys()):
                    lines.append(f"    {key:<{max_key_length}} = {self.sim_sources['def_Mod'][f'{key}']}\n")
                lines.append(end_line)

                for line in lines:
                    f1.write(line)
            time.sleep(0.8)
        # return True

    def __step3_check_newcase_namelist(self, newlib):
        check_state = 0

        general = newlib['general']
        model_key = "Sim_casename"
        timezone_key = "timezone"
        data_groupby_key = "data_groupby"
        dir_key = "root_dir"
        tim_res_key = "tim_res"

        # 这之后的要区分检查--------------------------------
        data_type_key = "data_type"
        fulllist_key = "fulllist"
        geo_res_key = "grid_res"
        suffix_key = "suffix"
        prefix_key = "prefix"
        syear_key = "syear"
        eyear_key = "eyear"

        if len(newlib[model_key]) < 1:
            st.error(f'{model_key} should be a string longer than one, please check {model_key}.',
                     icon="⚠")
            check_state += 1

        if newlib['Mod']:
            for key in [timezone_key, data_groupby_key, dir_key, tim_res_key]:
                if isinstance(general[key], str):
                    if len(general[key]) < 1:
                        st.error(f'{key} should be a string longer than one, please check {key}.',
                                 icon="⚠")
                        check_state += 1
                elif isinstance(general[key], float) | isinstance(general[key], int):
                    if general[key] < -12. or general[key] > 12.:
                        st.error(f'"please check {key}.', icon="⚠")
                        check_state += 1

            if general[data_type_key] == "Grid":
                if (general[geo_res_key] == 0.0):
                    st.error(f"Geo Resolution should be larger than zero when data_type is 'geo', please check.", icon="⚠")
                    check_state += 1
                elif (suffix_key in general) and (prefix_key in general):
                    if isinstance(general[suffix_key], str) | (isinstance(general[prefix_key], str)):
                        if len(general[suffix_key]) == 0 and len(general[prefix_key]) == 0:
                            st.error(f'"suffix or prefix should be a string longer than one, please check.', icon="⚠")
                            check_state += 1
                elif general[eyear_key] < general[syear_key]:
                    st.error(f" year should be larger than Start year, please check.", icon="⚠")
                    check_state += 1
            elif general[data_type_key] == "stn":
                if not general[fulllist_key]:
                    st.error(f"Fulllist should not be empty when data_type is 'stn'.", icon="⚠")
                    check_state += 1

            model_var = self.nl.read_namelist(general['model_namelist'])
            for selected_item in self.selected_items:
                if selected_item not in model_var.keys():
                    st.warning(f"{selected_item.replace('_', ' ')} not in {newlib['Mod']}, please make sure!", icon="⚠")

            if newlib['variables'] and newlib['info_list']:
                for variable in newlib['variables']:
                    warning = 0
                    if (suffix_key in newlib[variable]) and (prefix_key in newlib[variable]):
                        if isinstance(newlib[variable][suffix_key], str) | (isinstance(newlib[variable][prefix_key], str)):
                            if len(newlib[variable][suffix_key]) == 0 and len(newlib[variable][prefix_key]) == 0:
                                st.error(f'{variable} "suffix or prefix should be a string longer than one, please check.',
                                         icon="⚠")
                    for info in newlib['info_list']:
                        if isinstance(newlib[variable][info], str) and info not in [suffix_key, prefix_key]:
                            if len(newlib[variable][info]) == 0:
                                warning += 1
                    if warning > 0:
                        st.warning(f'{variable} exist warning, please check.', icon="⚠")



        else:
            st.error(f"Please select Model first.", icon="⚠")
            check_state += 1

        if check_state > 0:
            return False
        if check_state == 0:
            return True

    def step3_mange_simcases(self):
        if 'step3_remove' not in st.session_state:
            st.session_state['step3_remove'] = False
        sim_sources = self.nl.read_namelist('./GUI/Namelist_lib/Simulation_lib.nml')

        def get_cases(sim_sources):
            case_item = {}
            for item in sim_sources['general']['Case_lib']:
                case_item[item] = False

            st.subheader("Simulation Cases ....", divider=True)
            cols = itertools.cycle(st.columns(2))
            for item in case_item:
                col = next(cols)
                case_item[item] = col.checkbox(item, key=item,
                                               value=case_item[item])
            return [item for item, value in case_item.items() if value]

        cases = get_cases(sim_sources)

        def remove(cases):
            with st.spinner('Making namelist... Please wait.'):
                for case in cases:
                    if case in sim_sources['def_nml']:
                        if os.path.exists(sim_sources['def_nml'][case]):
                            os.remove(sim_sources['def_nml'][case])
                            st.code(f"Remove file: {sim_sources['def_nml'][case]}", wrap_lines=True)
                        else:
                            st.warning(f"{case} already removed or file doesn't exist!")
                    else:
                        st.warning(f"{case} already removed or file doesn't exist!")

                with open('./GUI/Namelist_lib/Simulation_lib.nml', 'w') as f1:
                    lines = []
                    end_line = "/\n\n\n"
                    lines.append("&general\n")
                    if isinstance(sim_sources['general']['Case_lib'], str): sim_sources['general'][
                        'Case_lib'] = [
                        sim_sources['general']['Case_lib']]
                    value = [case for case in sim_sources['general']['Case_lib'] if case not in cases]
                    if isinstance(value, str): value = [value]
                    lines.append(f"    Case_lib = {', '.join(value)}\n")
                    lines.append(end_line)

                    lines.append("&def_nml\n")
                    max_key_length = max(len(key) for key in sim_sources['def_nml'].keys())
                    for key, value in sim_sources['def_nml'].items():
                        if key not in cases:
                            lines.append(f"    {key:<{max_key_length}} = {value}\n")
                    lines.append(end_line)

                    lines.append("&def_Mod\n")
                    max_key_length = max(len(key) for key in sim_sources['def_Mod'].keys())
                    for key in list(sim_sources['def_Mod'].keys()):
                        lines.append(f"    {key:<{max_key_length}} = {sim_sources['def_Mod'][f'{key}']}\n")
                    lines.append(end_line)

                    for line in lines:
                        f1.write(line)
                    # time.sleep(2)
            for item in st.session_state.sim_data['general'].keys():
                for case in cases:
                    if case in st.session_state.sim_data['general'][item]:
                        st.session_state.sim_change['general'] = True
                st.session_state.sim_data['general'][item] = [value for value in st.session_state.sim_data['general'][item] if
                                                              value not in cases]

        def define_remove():
            st.session_state.step3_remove = True

        disable = False
        if not cases:
            disable = True

        remove_contain = st.container()

        if st.button('Remove cases', on_click=define_remove, help='Press to remove cases', disabled=disable):
            with remove_contain:
                remove(cases)
                sim_sources = self.nl.read_namelist('./GUI/Namelist_lib/Simulation_lib.nml')

        st.divider()

        def define_back(remove_contain, warn):
            if not disable:
                if not st.session_state.step3_remove:
                    with remove_contain:
                        remove(cases)
                st.session_state.step3_remove = False
            # else:
            # remove_contain.error('Remove file failed!')
            # time.sleep(0.8)
            st.session_state.step3_mange_cases = False

        st.button('Finish and back to select page', on_click=define_back, args=(remove_contain, 'remove_contain'),
                  help='Finish add and go back to select page')


class make_simulation(mange_simulation):
    def __init__(self, initial):
        self.author = "Qingchen Xu/xuqingchen23@163.com"
        self.classification = initial.classification()

        self.sim = initial.sim()
        self.evaluation_items = st.session_state.evaluation_items
        self.selected_items = [k for k, v in self.evaluation_items.items() if v]
        self.tittles = [k.replace('_', ' ') for k, v in self.evaluation_items.items() if v]
        self.initial = initial
        self.nl = NamelistReader()
        self.sim_sources = self.nl.read_namelist('./GUI/Namelist_lib/Simulation_lib.nml')
        self.Mod_variables_defination = os.path.join(st.session_state.openbench_path, 'nml', 'Mod_variables_defination')
        self.lib_path = os.path.join(st.session_state.openbench_path, 'nml', 'user')
        self.path_finder = FindPath()
        self.base_path = Path(st.session_state.openbench_path)

    def step3_set(self):

        if 'sim_change' not in st.session_state:
            st.session_state.sim_change = {'general': False}

        st.subheader(f'Select your simulation cases', divider=True)

        if 'add_mode' not in st.session_state:
            st.session_state['add_mod'] = False

        sim_general = self.sim['general']
        if st.session_state.sim_data['general']:
            sim_general = st.session_state.sim_data['general']

        def sim_data_change(key, editor_key):
            sim_general[key] = st.session_state[editor_key]
            st.session_state.sim_change['general'] = True

        for selected_item in self.selected_items:
            item = f"{selected_item}_sim_source"
            if item not in sim_general:
                sim_general[item] = []
            if isinstance(sim_general[item], str): sim_general[item] = [sim_general[item]]

            label_text = f"<span style='font-size: 20px;'>{selected_item.replace('_', ' ')} simulation cases ....</span>"
            st.markdown(f":blue[{label_text}]", unsafe_allow_html=True)
            if len(self.sim_sources['general']['Case_lib']) == 0:
                st.warning(
                    f"Sorry we didn't offer simulation data, please upload!")

            st.multiselect("simulation offered",
                           [value for value in self.sim_sources['general']['Case_lib']],
                           default=[value for value in sim_general[item] if value in self.sim_sources['general']['Case_lib']],
                           key=f"{item}_multi",
                           on_change=sim_data_change,
                           args=(item, f"{item}_multi"),
                           placeholder="Choose an option",
                           label_visibility="collapsed")

        st.session_state.step3_set_check = self.__step3_setcheck(sim_general)

        sources = list(set([value for key in self.selected_items for value in sim_general[f"{key}_sim_source"] if value]))
        st.session_state.sim_data['def_nml'] = {}
        for source in sources:
            st.session_state.sim_data['def_nml'][source] = self.sim_sources['def_nml'][source]
            st.session_state.sim_change[source] = False

        keys = st.session_state.sim_data.keys()
        from difflib import ndiff
        diff = list(ndiff(list(keys), sources))
        for source in diff:
            if source.startswith('- ') and source not in ['- general', '- def_nml']:
                del st.session_state.sim_data[source.split(' ')[1]]

        formatted_keys = " \n".join(
            f'{key.replace("_", " ")}: {", ".join(value for value in sim_general[f"{key}_sim_source"] if value)}' for key in self.selected_items)
        sourced_key = " \n".join(f"{source}: {self.sim_sources['def_nml'][source]}" for source in sources)
        st.code(f'''{formatted_keys}\n\n{sourced_key}''', language="shell", line_numbers=True, wrap_lines=True)

        #

        col1, col, col3 = st.columns(3)

        def define_new_simnml():
            st.session_state.step3_make_newnamelist = True

        col1.button('Add new simulation namelist', on_click=define_new_simnml)

        def define_clear_cases():
            st.session_state.step3_mange_cases = True

        col3.button('Manage simulation cases', on_click=define_clear_cases)

        st.session_state.sim_data['general'] = sim_general

        st.divider()

        def define_step1():
            st.session_state.step3_set = False

        def define_step2():
            st.session_state.step3_make = True

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Refernece page')
        with col4:
            st.button('Next step :soon: ', on_click=define_step2, help='Go to making page')

    def __step3_setcheck(self, general_sim):
        check_state = 0

        for selected_item in self.selected_items:
            key = f"{selected_item}_sim_source"
            if general_sim[key] is None or len(general_sim[key]) == 0:
                st.warning(f'Please set at least one case in {key.replace("_", " ")}!', icon="⚠")
                check_state += 1
            if selected_item not in st.session_state.step3_errorlist:
                st.session_state.step3_errorlist[selected_item] = []

            if check_state > 0:
                st.session_state.step3_errorlist[selected_item].append(1)
                st.session_state.step3_errorlist[selected_item] = list(
                    np.unique(st.session_state.step3_errorlist[selected_item]))
            if check_state == 0:
                if (selected_item in st.session_state.step3_errorlist) & (
                        1 in st.session_state.step3_errorlist[selected_item]):
                    st.session_state.step3_errorlist[selected_item] = list(
                        filter(lambda x: x != 1, st.session_state.step3_errorlist[selected_item]))
                    st.session_state.step3_errorlist[selected_item] = list(
                        np.unique(st.session_state.step3_errorlist[selected_item]))

        if check_state > 0:
            return False
        if check_state == 0:
            return True

    def step3_make(self):
        sim_general = st.session_state.sim_data['general']
        def_nml = st.session_state.sim_data['def_nml']

        if sim_general and def_nml:
            st.session_state.step3_check = []
            for i, (source, path), tab in zip(range(len(def_nml)), def_nml.items(), st.tabs(def_nml.keys())):
                if source not in st.session_state.sim_data:
                    st.session_state.sim_data[source] = self.nl.read_namelist(path)
                    st.session_state.sim_data[source] = self.nl.Update_namelist(st.session_state.sim_data[source])
                tab.subheader(f':blue[{source} Simulation checking ....]', divider=True)
                with tab:
                    self.__step3_make_sim_info(i, source, st.session_state.sim_data[source], path, sim_general)  # self.ref

            if all(st.session_state.step3_check):
                st.session_state.step3_make_check = True
            else:
                st.session_state.step3_make_check = False
        else:
            st.error('Please select your case first!')
            st.session_state.step3_make_check = False
            st.divider()

        def define_step1():
            st.session_state.step3_make = False

        def define_step2():
            st.session_state.step3_nml = True

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Simulation set page')
        with col4:
            st.button('Next step :soon: ', on_click=define_step2, help='Go to Simulation nml page')

    def __step3_make_sim_info(self, i, source, source_lib, file, sim_general):
        def sim_editor_change(key, editor_key, source):
            source_lib[key][editor_key] = st.session_state[f"{source}_{key}_{editor_key}"]
            st.session_state.sim_change[source] = True

        def set_data_type(compare_tres_value):
            my_list = ['stn', 'grid']
            index = my_list.index(compare_tres_value.lower())
            return index

        def set_data_groupby(compare_tres_value):
            my_list = ['hour', 'day', 'month', 'year', 'single']
            index = my_list.index(compare_tres_value.lower())
            return index

        import itertools
        with st.container(height=None, border=True):
            key = 'general'
            col1, col2, col3 = st.columns(3)
            source_lib[key]['data_type'] = col1.selectbox(f'{i}. Set Data type: ',
                                                          options=('stn', 'grid'),
                                                          index=set_data_type(source_lib[key]['data_type']),
                                                          placeholder=f"Set your Simulation Data type (default={source_lib[key]['data_type']})...")
            source_lib[key]['tim_res'] = col2.selectbox(f'{i}. Set Time Resolution: ',
                                                        options=('hour', 'day', 'month', 'year'),
                                                        index=set_data_groupby(source_lib[key]['tim_res']),
                                                        placeholder=f"Set your Simulation Time Resolution (default={source_lib[key]['tim_res']})...")
            source_lib[key]['data_groupby'] = col3.selectbox(f'{i}. Set Data groupby: ',
                                                             options=('hour', 'day', 'month', 'year', 'single'),
                                                             index=set_data_groupby(source_lib[key]['data_groupby']),
                                                             placeholder=f"Set your Simulation Data groupby (default={source_lib[key]['data_groupby']})...")

            cols = itertools.cycle(st.columns(3))
            for item in source_lib[key].keys():
                if item not in ["model_namelist", "root_dir", 'data_type', 'tim_res', 'data_groupby']:
                    col = next(cols)
                    if source_lib[key]['data_type'] == 'stn':
                        if item == 'fulllist':
                            if "fulllist" not in source_lib[key]: source_lib[key][f"fulllist"] = None
                            if not source_lib[key][f"fulllist"]: source_lib[key][f"fulllist"] = None
                            source_lib[key][f"fulllist"] = self.path_finder.get_file(source_lib[key][f"fulllist"], f"{source}_{key}_fulllist",
                                                                                     'csv', ['sim_change', source])
                            st.code(f"Set Fulllist File: {source_lib[key][f'fulllist']}", language='shell', wrap_lines=True)
                        elif item in ['prefix', 'suffix', 'grid_res', 'syear', 'eyear']:
                            source_lib[key][item] = ''
                        elif item in ['timezone']:
                            source_lib[key][item] = col.number_input(f"{i}. Set {item}: ",
                                                                     value=float(source_lib[key][item]),
                                                                     key=f"{source}_{key}_{item}",
                                                                     on_change=sim_editor_change,
                                                                     args=(key, item, source),
                                                                     placeholder=f"Set your Simulation {item}...")
                    else:
                        if item in ['prefix', 'suffix']:
                            source_lib[key][item] = col.text_input(f'{i}. Set {item}: ',
                                                                   value=source_lib[key][item],
                                                                   key=f"{source}_{key}_{item}",
                                                                   on_change=sim_editor_change,
                                                                   args=(key, item, source),
                                                                   placeholder=f"Set your Simulation {item}...")

                        elif item in ['syear', 'eyear']:
                            source_lib[key][item] = col.number_input(f"{i}. Set {item}:",
                                                                     format='%04d', step=int(1),
                                                                     value=source_lib[key][item],
                                                                     key=f"{source}_{key}_{item}",
                                                                     on_change=sim_editor_change,
                                                                     args=(key, item, source),
                                                                     placeholder=f"Set your Simulation {item}...")

            if "root_dir" not in source_lib[key]: source_lib[key][f"root_dir"] = '/'
            if not source_lib[key][f"root_dir"]: source_lib[key][f"root_dir"] = '/'
            source_lib[key][f"root_dir"] = self.path_finder.find_path(source_lib[key][f"root_dir"],
                                                                      f"{source}_general_root_dir",
                                                                      ['sim_change', source])
            st.code(f"Set Data Dictionary: {source_lib[key][f'root_dir']}", language='shell', wrap_lines=True)

            for key, values in source_lib.items():
                if key != 'general' and key in self.selected_items:
                    if source in sim_general[f'{key}_sim_source'] and len(source_lib[key]) > 0:
                        st.divider()
                        st.write(f'##### :blue[{key.replace("_", " ")}]')
                        cols = itertools.cycle(st.columns(2))
                        for info in sorted(source_lib[key].keys(), key=lambda x: (x == "sub_dir", x)):
                            col = next(cols)
                            if info == 'sub_dir':
                                with col:
                                    if info not in source_lib[key]: source_lib[key][info] = ''
                                    if not source_lib[key][info]: source_lib[key][info] = ''
                                    source_lib[key][info] = self.path_finder.find_subdirectories(source_lib[key][info],
                                                                                                 f"{source}_general_root_dir",
                                                                                                 f"{source}_{key}_{info}",
                                                                                                 ['sim_change', source])
                                    st.code(f"Sub-Dir: {source_lib[key][f'sub_dir']}", language='shell', wrap_lines=True)
                            else:
                                source_lib[key][info] = col.text_input(
                                    f'{i}. Set {info}: ',
                                    value=source_lib[key][info],
                                    key=f"{source}_{key}_{info}",
                                    on_change=sim_editor_change,
                                    args=(key, info, source),
                                    placeholder=f"Set your Reference Dictionary...")

            st.session_state.step3_check.append(self.__step3_makecheck(source_lib, source))

    def __step3_makecheck(self, source_lib, source):
        error_state = 0

        info_list = ['varname', 'varunit', ]
        general = source_lib['general']
        model_key = "model_namelist"
        timezone_key = "timezone"
        data_groupby_key = "data_groupby"
        dir_key = "root_dir"
        tim_res_key = "tim_res"

        # 这之后的要区分检查--------------------------------
        data_type_key = "data_type"
        fulllist_key = "fulllist"
        geo_res_key = "grid_res"
        suffix_key = "suffix"
        prefix_key = "prefix"
        syear_key = "syear"
        eyear_key = "eyear"

        if len(general[model_key]) <= 1:
            st.error(f'{model_key} should be a string longer than one, please check {model_key}.',
                     icon="⚠")
            error_state += 1

        for key in [timezone_key, data_groupby_key, dir_key, tim_res_key]:
            if isinstance(general[key], str):
                if len(general[key]) <= 1:
                    st.error(f'{key} should be a string longer than one, please check {key}.',
                             icon="⚠")
                    error_state += 1
            elif isinstance(general[key], float) | isinstance(general[key], int):
                if general[key] < -12. or general[key] > 12.:
                    st.error(f'"please check {key}.', icon="⚠")
                    error_state += 1

        if general[data_type_key] == "Grid":
            if (general[geo_res_key] == 0.0):
                st.error(f"Geo Resolution should be larger than zero when data_type is 'geo', please check.", icon="⚠")
                error_state += 1
            elif (isinstance(general[suffix_key], str)) | (isinstance(general[prefix_key], str)):
                if (len(general[suffix_key]) == 0) & (len(general[prefix_key]) == 0):
                    st.error(f'"suffix or prefix should be a string longer than one, please check.', icon="⚠")
                    error_state += 1
            elif general[eyear_key] < general[syear_key]:
                st.error(f" year should be larger than Start year, please check.", icon="⚠")
                error_state += 1
            for key in [syear_key, eyear_key]:
                if isinstance(general[key], int):
                    st.error(f'"please check {key}.', icon="⚠")
                    error_state += 1

        elif general[data_type_key] == "stn":
            if not general[fulllist_key]:
                st.error(f"Fulllist should not be empty when data_type is 'stn'.", icon="⚠")
                error_state += 1

        # model_var = self.nl.read_namelist(general['model_namelist'])
        # for selected_item in self.selected_items:
        #     if selected_item not in model_var.keys():
        #         st.warning(f"{selected_item.replace('_', ' ')} not in {general[model_key]}, please make sure!", icon="⚠")

        if source not in st.session_state.step3_errorlist:
            st.session_state.step3_errorlist[source] = []
        if error_state > 0:
            st.session_state.step3_errorlist[source].append(2)
            st.session_state.step3_errorlist[source] = list(np.unique(st.session_state.step3_errorlist[source]))
            return False
        if error_state == 0:
            if (source in st.session_state.step3_errorlist) & (2 in st.session_state.step3_errorlist[source]):
                st.session_state.step3_errorlist[source] = list(
                    filter(lambda x: x != 2, st.session_state.step3_errorlist[source]))
                st.session_state.step3_errorlist[source] = list(np.unique(st.session_state.step3_errorlist[source]))
            return True

    def step3_sim_nml(self):

        step3_disable = False

        Mod = self.sim_sources['def_Mod']
        if st.session_state.step3_set_check & st.session_state.step3_make_check:
            for source, path in st.session_state.sim_data['def_nml'].items():
                source_lib = st.session_state.sim_data[source]
                st.subheader(source, divider=True)
                model_info = ''
                for key, value in Mod.items():
                    if source_lib["general"][f"model_namelist"] == value:
                        model_info = f'Model: {key}'
                path_info = f'Root Dictionary: {source_lib["general"][f"root_dir"]}'
                key = 'general'
                if source_lib[key]['data_type'] == 'stn':
                    path_info = path_info + f'\nFulllist File: {source_lib["general"][f"fulllist"]}'
                st.code(f'''{model_info}\n{path_info}''', language='shell', line_numbers=True, wrap_lines=True)
            st.session_state.step3_sim_nml = False
            st.session_state.step3_sim_check = True
        else:
            step3_disable = True
            if not st.session_state.step3_set_check:
                formatted_keys = ", ".join(
                    key.replace('_', ' ') for key, value in st.session_state.step3_errorlist.items() if 1 in value)
                st.error(
                    f'There exist error in set page, please check {formatted_keys} first! Set your simulation case.',
                    icon="🚨")
            if not st.session_state.step3_make_check:
                formatted_keys = ", ".join(
                    key.replace('_', ' ') for key, value in st.session_state.step3_errorlist.items() if 2 in value)
                st.error(f'There exist error in Making page, please check {formatted_keys} first!', icon="🚨")
            st.session_state.step3_sim_nml = False
            st.session_state.step3_sim_check = False

        def write_nml(nml_dict, output_file):
            """
            将字典数据重新写回 .nml 文件。
            """
            with open(output_file, 'w') as f:
                # 确保 'general' 部分总是第一个
                if 'general' in nml_dict:
                    f.write(f'&general\n')
                    max_key_length = max(len(key) for key in nml_dict['general'].keys())
                    for key, value in nml_dict['general'].items():
                        f.write(f'  {key:<{max_key_length}} = {value}\n')
                    f.write('/\n\n')

                # 写入其他部分
                for section, variables in nml_dict.items():
                    if section == 'general':
                        continue  # 'general' 已经处理过了
                    f.write(f'&{section}\n')
                    for key, value in variables.items():
                        if key != 'sub_dir':
                            f.write(f'  {key} = {value}\n')
                        else:
                            if value is None:
                                f.write(f'  {key} = \n')
                            else:
                                f.write(f'  {key} = {value}\n')
                    f.write('/\n\n')
            del f

        def make():
            for key, value in st.session_state.sim_change.items():
                if key == 'general' and value:
                    if st.session_state.step3_sim_check & (not st.session_state.step3_sim_nml):
                        st.session_state.step3_sim_nml = self.__step3_make_sim_namelist(
                            st.session_state.generals['simulation_nml'], self.selected_items, st.session_state.sim_data)
                    if st.session_state.step3_sim_nml:
                        st.success("😉 Make file successfully!!!")
                        st.session_state.sim_change[key] = False
                elif key != 'general' and value:
                    file = st.session_state.sim_data['def_nml'][key]
                    source_lib = st.session_state.sim_data[key]
                    write_nml(source_lib, file)
                    st.session_state.sim_change[key] = False

        def define_step1():
            st.session_state.step3_nml = False

        def define_step2(make_contain, smake):
            if not st.session_state.step3_sim_check:
                st.session_state.step4_set = False
            else:
                with make_contain:
                    if any(v for v in st.session_state.sim_change.values()):
                        make()
                if st.session_state.get('switch_button1', True):
                    st.session_state.step4_set = True
                    st.session_state.step3 = True
                    st.session_state.switch_button1_onclick = +1
                    st.session_state['menu_option'] = (switch_button_index(st.session_state.selected) + 1) % 5

        def switch_button_index(select):
            my_list = ["Home", "Evaluation", "Running", 'Visualization', 'Statistics']
            index = my_list.index(select)
            return index

        st.divider()
        make_contain = st.container()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Simulation Making page')
        with col4:
            st.button('Next step :soon: ', help='Press to go to Run page', on_click=define_step2,
                      args=(make_contain, 'make'), disabled=step3_disable,
                      key='switch_button1')

    def __step3_make_sim_namelist(self, file_path, selected_items, sim_data):
        general = sim_data['general']
        def_nml = sim_data['def_nml']

        with st.spinner('Making namelist... Please wait.'):
            if st.session_state.step3_sim_check:
                with open(file_path, 'w') as f:
                    lines = []
                    end_line = "/\n\n\n"

                    lines.append("&general\n")
                    max_key_length = max(len(f"{key}") for key in general.keys())
                    for item in selected_items:
                        key = f"{item}_sim_source"
                        lines.append(f"    {key:<{max_key_length}} = {','.join(general[f'{item}_sim_source'])}\n")
                    lines.append(end_line)

                    lines.append("&def_nml\n")
                    max_key_length = max(len(key) for key in def_nml.keys())
                    for key, value in def_nml.items():
                        if Path(value).is_relative_to(self.base_path):
                            value = './' + os.path.relpath(value, self.base_path)
                        lines.append(f"    {key:<{max_key_length}} = {value}\n")
                    lines.append(end_line)
                    for line in lines:
                        f.write(line)
                    time.sleep(2)
                    return True
