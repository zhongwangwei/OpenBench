# -*- coding: utf-8 -*-
import glob, shutil
import os
import sys
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
from pathlib import Path


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


class mange_reference:
    def __init__(self):
        self.author = "Qingchen Xu/xuqingchen0@gmail.com"

    def step2_make_new_refnml(self):
        if 'step2_add_nml' not in st.session_state:
            st.session_state.step2_add_nml = False

        st.markdown(f"""
        <div style="font-size:22px; font-weight:bold; color:#68838B; border-bottom:3px solid #68838B; padding: 5px;">
            Set reference form 👇
        </div>""", unsafe_allow_html=True)
        st.write(' ')
        form = st.radio("#### Set reference form 👇", ["Composite", "Single"], key="ref_form",
                        label_visibility='collapsed',
                        horizontal=True,
                        captions=['If reference has multi-variables', 'If reference has only one variable'])
        st.markdown(f"""
        <div style="font-size:22px; font-weight:bold; color:#68838B; border-bottom:3px solid #68838B; padding: 5px;">
            Set reference file 👇
        </div>""", unsafe_allow_html=True)
        st.write(' ')
        file = st.radio("Set reference file 👇",
                        ['Composite', 'Crop', 'Dam', 'Ecosystem', 'Energy', 'Forcing', 'Hydrology', 'Lake', 'River', 'Urban'],
                        key="ref_savefile", label_visibility='collapsed',
                        horizontal=True)
        ref_save_path = os.path.join(self.lib_path, file)
        st.divider()

        newlib = {}

        def variables_change(key, editor_key):
            newlib[key] = st.session_state[editor_key]

        def ref_main_change(key, editor_key):
            if 'general' not in newlib:
                newlib['general'] = {}
            newlib['general'][key] = st.session_state[editor_key]

        def ref_info_change(key, editor_key):
            newlib[key] = st.session_state[f"{key}_{editor_key}"]

        def get_var(col):
            if 'reflib_item' not in st.session_state:
                st.session_state['reflib_item'] = self.initial.evaluation_items()
            Evaluation_Items = st.session_state['reflib_item']

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
                # st.subheader(":blue[]", divider=True)
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

        col1, col2 = st.columns((1, 2))
        newlib['Ref_libname'] = col1.text_input(f'Reference lib name: ', value='',
                                                key=f"Ref_libname",
                                                on_change=variables_change,
                                                args=(f"Ref_libname", 'Ref_libname'),
                                                placeholder=f"Set your Reference lib...")
        if st.session_state['ref_form'] == 'Composite':
            info_list = ['sub_dir', 'varname', 'varunit', 'prefix', 'suffix', 'syear', 'eyear']
            newlib['variables'] = get_var(col2)
            if len(newlib['Ref_libname']) > 0 and newlib['variables']:
                newlib['general'] = {}
                if 'root_dir' not in newlib['general']: newlib['general']['root_dir'] = './data/'
                if not newlib['general']['root_dir']: newlib['general']['root_dir'] = './data/'
                find_path = self.path_finder.find_path(newlib['general']['root_dir'], f"newlib_{newlib['Ref_libname']}_root_dir",
                                                       [None, None])
                st.code(f"Dictionary: {find_path}", language='shell', wrap_lines=True)
                newlib['general']['root_dir'] = find_path

                col1, col2, col3 = st.columns(3)
                newlib['general']['timezone'] = col1.number_input(f"Set Time zone: ",
                                                                  value=0.0,
                                                                  key=f"new_lib_timezone",
                                                                  on_change=ref_main_change,
                                                                  args=(f"timezone", 'new_lib_timezone'),
                                                                  min_value=-12.0,
                                                                  max_value=12.0)
                newlib['general']['data_type'] = col2.selectbox(f'Set Data type: ',
                                                                options=('stn', 'Grid'),
                                                                index=1,
                                                                key=f"new_lib_data_type",
                                                                on_change=ref_main_change,
                                                                args=(f"data_type", 'new_lib_data_type'),
                                                                placeholder=f"Set your Reference Data type...")
                newlib['general']['data_groupby'] = col3.selectbox(f'Set Data groupby: ',
                                                                   options=('hour', 'day', 'month', 'year', 'single'),
                                                                   index=4,
                                                                   key=f"new_lib_data_groupby",
                                                                   on_change=ref_main_change,
                                                                   args=(f"data_groupby", 'new_lib_data_groupby'),
                                                                   placeholder=f"Set your Reference Data groupby...")
                newlib['general']['tim_res'] = col1.selectbox(f'Set Time Resolution: ',
                                                              options=('hour', 'day', 'month', 'year'),
                                                              index=0,
                                                              key=f"new_lib_tim_res",
                                                              on_change=ref_main_change,
                                                              args=(f"tim_res", 'new_lib_tim_res'),
                                                              placeholder=f"Set your Reference Time Resolution ...")
                if newlib['general']['data_type'] == 'Grid':
                    newlib['general']['grid_res'] = col2.number_input(f"Set Geo Resolution: ",
                                                                      value=0.5,
                                                                      min_value=0.0,
                                                                      key=f"new_lib_grid_res",
                                                                      on_change=ref_main_change,
                                                                      args=(f"grid_res", 'new_lib_grid_res'),
                                                                      placeholder="Set your Reference Geo Resolution...")
                    year = st.toggle(f'Reference {newlib["Ref_libname"]} share the same start and end year?',
                                     value=True)
                    if year:
                        info_list = ['sub_dir', 'varname', 'varunit', 'prefix', 'suffix']
                        col1, col2, col3 = st.columns(3)
                        newlib['general']['syear'] = col1.number_input(f"Set Start year: ",
                                                                       value=2000,
                                                                       format='%04d', step=int(1),
                                                                       key=f"new_lib_syear",
                                                                       on_change=ref_main_change,
                                                                       args=(f"syear", 'new_lib_syear'),
                                                                       placeholder="Set your Reference Start year...")
                        newlib['general']['eyear'] = col2.number_input(f"Set End year: ",
                                                                       value=2001,
                                                                       format='%04d', step=int(1),
                                                                       key=f"new_lib_eyear",
                                                                       on_change=ref_main_change,
                                                                       args=(f"eyear", 'new_lib_eyear'),
                                                                       placeholder="Set your Reference End year...")
                else:
                    info_list = ['varname', 'varunit']
                    if 'fulllist' not in newlib['general']: newlib['general']['fulllist'] = None
                    if not newlib['general'][f"fulllist"]: newlib['general'][f"fulllist"] = None
                    newlib['general'][f"fulllist"] = self.path_finder.get_file(newlib['general'][f"fulllist"],
                                                                               f"new_lib_fulllist",
                                                                               'csv', [None, None])
                    st.code(f"Set Fulllist File: {newlib['general'][f'fulllist']}", language='shell', wrap_lines=True)

                    newlib['general']['grid_res'] = ''
                    newlib['general']['syear'] = ''
                    newlib['general']['eyear'] = ''
                st.write('##### :orange[Add info]')
                newlib['info_list'] = st.multiselect("Add info", info_list, default=['varname', 'varunit'],
                                                     key=f"variables_info_list",
                                                     on_change=variables_change,
                                                     args=('info_list', f"variables_info_list"),
                                                     placeholder="Choose an option",
                                                     label_visibility="collapsed")

                info_lists = {'sub_dir': {'title': 'Set Data Sub-Dictionary', 'value': ''},
                              'varname': {'title': 'Set varname', 'value': ''},
                              'varunit': {'title': 'Set varunit', 'value': ''},
                              'prefix': {'title': 'Set prefix', 'value': ''},
                              'suffix': {'title': 'Set suffix', 'value': ''},
                              'syear': {'title': 'Set syear', 'value': 2000},
                              'eyear': {'title': 'Set eyear', 'value': 2001}}
                if newlib['variables']:
                    for var in self.evaluation_items:
                        if var in newlib['variables']:
                            newlib[var] = {}
                        if var in newlib and var not in newlib['variables']:
                            del newlib[var]

                    import itertools
                    for variable in newlib['variables']:
                        with st.container(height=None, border=True):
                            st.write(f"##### :blue[{variable.replace('_', ' ')}]")
                            cols = itertools.cycle(st.columns(3))
                            for info in st.session_state['variables_info_list']:
                                col = next(cols)
                                if info == 'sub_dir':
                                    with  st.container():
                                        if 'sub_dir' not in newlib[variable]: newlib[variable][info] = info_lists[info]['value']
                                        if not newlib[variable][info]: newlib[variable][info] = ''
                                        newlib[variable][info] = self.path_finder.find_subdirectories(newlib[variable][info],
                                                                                                      f"newlib_{newlib['Ref_libname']}_root_dir",
                                                                                                      f"newlib_{variable}_{info}",
                                                                                                      [None, None])
                                        st.code(f"Sub-Dir: {newlib[variable][info]}", language='shell', wrap_lines=True)
                                else:
                                    newlib[variable][info] = col.text_input(info_lists[info]['title'],
                                                                            value=info_lists[info]['value'],
                                                                            key=f"{variable}_{info}",
                                                                            on_change=ref_info_change,
                                                                            args=(variable, info),
                                                                            placeholder=f"Set your Reference Info...")

        else:
            info_list = ['varname', 'varunit', 'prefix', 'suffix']
            newlib['variables'] = col2.selectbox("Variable selected",
                                                 [e.replace('_', ' ') for e in self.evaluation_items], index=None,
                                                 key=f"variable_select",
                                                 on_change=variables_change,
                                                 args=('variables', f"variable_select"),
                                                 placeholder="Choose an option",
                                                 label_visibility="visible")

            if newlib['Ref_libname'] and st.session_state['variable_select']:
                newlib['general'] = {}
                if 'root_dir' not in newlib['general']: newlib['general']['root_dir'] = os.path.abspath('./data/')
                if not newlib['general']['root_dir']: newlib['general']['root_dir'] = os.path.abspath('./data/')
                find_path = self.path_finder.find_path(newlib['general']['root_dir'], f"newlib_{newlib['Ref_libname']}_root_dir",
                                                       [None, None])
                st.code(f"Dictionary: {find_path}", language='shell', wrap_lines=True)
                newlib['general']['root_dir'] = find_path

                col1, col2, col3 = st.columns(3)
                newlib['general']['timezone'] = col1.number_input(f"Set Time zone: ",
                                                                  value=0.0,
                                                                  key=f"new_lib_timezone",
                                                                  on_change=ref_main_change,
                                                                  args=(f"timezone", 'new_lib_timezone'),
                                                                  min_value=-12.0,
                                                                  max_value=12.0)
                newlib['general']['data_type'] = col2.selectbox(f'Set Data type: ',
                                                                options=('stn', 'Grid'),
                                                                index=1,
                                                                key=f"new_lib_data_type",
                                                                on_change=ref_main_change,
                                                                args=(f"data_type", 'new_lib_data_type'),
                                                                placeholder=f"Set your Reference Data type...")
                newlib['general']['data_groupby'] = col3.selectbox(f'Set Data groupby: ',
                                                                   options=('hour', 'day', 'month', 'year', 'single'),
                                                                   index=4,
                                                                   key=f"new_lib_data_groupby",
                                                                   on_change=ref_main_change,
                                                                   args=(f"data_groupby", 'new_lib_data_groupby'),
                                                                   placeholder=f"Set your Reference Data groupby...")
                newlib['general']['tim_res'] = col1.selectbox(f'Set Time Resolution: ',
                                                              options=('hour', 'day', 'month', 'year'),
                                                              index=0,
                                                              key=f"new_lib_tim_res",
                                                              on_change=ref_main_change,
                                                              args=(f"tim_res", 'new_lib_tim_res'),
                                                              placeholder=f"Set your Reference Time Resolution ...")
                if newlib['general']['data_type'] == 'Grid':
                    newlib['general']['grid_res'] = col2.number_input(f"Set Geo Resolution: ",
                                                                      value=0.5,
                                                                      min_value=0.0,
                                                                      key=f"new_lib_grid_res",
                                                                      on_change=ref_main_change,
                                                                      args=(f"grid_res", 'new_lib_grid_res'),
                                                                      placeholder="Set your Reference Geo Resolution...")
                    newlib['general']['syear'] = col1.number_input(f"Set Start year: ",
                                                                   value=2000,
                                                                   format='%04d', step=int(1),
                                                                   key=f"new_lib_syear",
                                                                   on_change=ref_main_change,
                                                                   args=(f"syear", 'new_lib_syear'),
                                                                   placeholder="Set your Reference Start year...")
                    newlib['general']['eyear'] = col2.number_input(f"Set End year: ",
                                                                   value=2001,
                                                                   format='%04d', step=int(1),
                                                                   key=f"new_lib_eyear",
                                                                   on_change=ref_main_change,
                                                                   args=(f"eyear", 'new_lib_eyear'),
                                                                   placeholder="Set your Reference End year...")
                else:
                    info_list = ['varname', 'varunit']
                    if 'fulllist' not in newlib['general']: newlib['general']['fulllist'] = None
                    if not newlib['general'][f"fulllist"]: newlib['general'][f"fulllist"] = None
                    newlib['general'][f"fulllist"] = self.path_finder.get_file(newlib['general'][f"fulllist"],
                                                                               f"new_lib_fulllist",
                                                                               'csv', [None, None])
                    st.code(f"Set Fulllist File: {newlib['general'][f'fulllist']}", language='shell', wrap_lines=True)
                    newlib['general']['grid_res'] = ''
                    newlib['general']['syear'] = ''
                    newlib['general']['eyear'] = ''
                st.write('##### :orange[Add info]')
                newlib['info_list'] = st.multiselect("Add info", info_list, default=info_list,
                                                     key=f"variables_info_list",
                                                     on_change=variables_change,
                                                     args=('info_list', f"variables_info_list"),
                                                     placeholder="Choose an option",
                                                     label_visibility="collapsed")

                info_lists = {'sub_dir': {'title': 'Set Data Sub-Dictionary', 'value': ''},
                              'varname': {'title': 'Set varname', 'value': ''},
                              'varunit': {'title': 'Set varunit', 'value': ''},
                              'prefix': {'title': 'Set prefix', 'value': ''},
                              'suffix': {'title': 'Set suffix', 'value': ''},
                              'syear': {'title': 'Set syear', 'value': 2000},
                              'eyear': {'title': 'Set eyear', 'value': 2001}}

                for var in self.evaluation_items:
                    if var.replace('_', ' ') in newlib['variables']:
                        newlib[var] = {}
                    if var in newlib and var.replace('_', ' ') not in newlib['variables']:
                        del newlib[var]

                import itertools
                variable = newlib['variables'].replace(' ', '_')
                with st.container(height=None, border=True):
                    st.write(f"##### :blue[{variable.replace('_', ' ')}]")
                    cols = itertools.cycle(st.columns(3))
                    for info in st.session_state['variables_info_list']:
                        col = next(cols)
                        newlib[variable][info] = col.text_input(info_lists[info]['title'],
                                                                value=info_lists[info]['value'],
                                                                key=f"{variable}_{info}",
                                                                on_change=ref_info_change,
                                                                args=(variable, info),
                                                                placeholder=f"Set your Reference Fulllist file...")

        disable = False

        if not newlib['Ref_libname']:
            st.error('Please input your Reference lib name First!')
            disable = True
        elif isinstance(newlib['variables'], list):
            if len(newlib['variables']) == 0:
                st.error('Please input your Reference lib variables First!')
                disable = True
        elif newlib['variables'] is None:
            st.error('Please input your Reference lib variables First!')
            disable = True
        else:
            if not newlib['general']['root_dir']:
                st.error('Please input your Reference Dictionary First!')
                disable = True
            elif newlib['general']['data_type'] == 'stn' and not newlib['general'][f"fulllist"]:
                st.error('Please input your Reference fulllist First!')
                disable = True

        def define_add():
            st.session_state.step2_add_nml = True

        col1, col2, col3 = st.columns(3)
        if col1.button('Make namelist', on_click=define_add, help='Press this button to add new reference namelist', disabled=disable):
            self.__step2_make_ref_lib_namelist(ref_save_path, newlib, form)
            st.success("😉 Make file successfully!!! \n Please press to Next step")

        st.divider()

        def define_back(make_contain, warn):
            if not disable:
                if not st.session_state.step2_add_nml:
                    with make_contain:
                        self.__step2_make_ref_lib_namelist(ref_save_path, newlib, form)
                    make_contain.success("😉 Make file successfully!!! \n Please press to Next step")
                del_vars = ['reflib_item']
                for del_var in del_vars:
                    del st.session_state[del_var]
                st.session_state.step2_add_nml = False

            else:
                make_contain.warning('Making file failed or No file is processing!')
                time.sleep(0.8)
            st.session_state.step2_make_newnamelist = False

        make_contain = st.container()
        st.button('Finish and back to select page', on_click=define_back, args=(make_contain, 'make_contain'),
                  help='Finish add and go back to select page')

    def __step2_make_ref_lib_namelist(self, file_path, newlib, fname):
        """
        Write a namelist from a text file.

        Args:
            file_path (str): Path to the text file.

        """
        with st.spinner('Making namelist... Please wait.'):
            with open(file_path + f'/{newlib["Ref_libname"]}.nml', 'w') as f:
                lines = []
                end_line = "/\n\n\n"

                lines.append("&general\n")
                max_key_length = max(len(key) for key in newlib['general'].keys())
                for key in list(newlib['general'].keys()):
                    lines.append(f"    {key:<{max_key_length}} = {newlib['general'][f'{key}']}\n")
                lines.append(end_line)

                max_key_length = max(len(key) for key in newlib['info_list'])
                if isinstance(newlib['variables'], str): newlib['variables'] = [newlib['variables']]
                for variable in newlib['variables']:
                    variable = variable.replace(' ', '_')
                    lines.append(f"&{variable}\n")
                    for info in newlib['info_list']:
                        lines.append(f"    {info:<{max_key_length}} = {newlib[variable][info]}\n")
                    lines.append(end_line)
                for line in lines:
                    f.write(line)
                time.sleep(2)

            if fname == 'Composite':
                for variable in newlib['variables']:
                    variable = variable.replace(' ', '_')
                    if isinstance(self.ref_sources['general'][variable], str): self.ref_sources['general'][variable] = [
                        self.ref_sources['general'][variable]]
                    if newlib["Ref_libname"] not in self.ref_sources['general'][variable]:
                        self.ref_sources['general'][variable].append(newlib["Ref_libname"])
            else:
                variable = variable.replace(' ', '_')
                if isinstance(self.ref_sources['general'][variable], str): self.ref_sources['general'][variable] = [
                    self.ref_sources['general'][variable]]
                if newlib["Ref_libname"] not in self.ref_sources['general'][variable]:
                    self.ref_sources['general'][variable].append(newlib["Ref_libname"])

            self.ref_sources['def_nml'][newlib["Ref_libname"]] = Path(
                os.path.relpath(os.path.join(file_path + f'{newlib["Ref_libname"]}.nml'), self.base_path))

            with open('./GUI/Namelist_lib/Reference_lib.nml', 'w') as f1:
                lines = []
                end_line = "/\n\n\n"
                lines.append("&general\n")
                max_key_length = max(len(key) for key in self.ref_sources['general'].keys())
                for key in list(self.ref_sources['general'].keys()):
                    value = self.ref_sources['general'][f'{key}']
                    if isinstance(value, str): value = [value]
                    lines.append(f"    {key:<{max_key_length}} = {', '.join(value)}\n")
                lines.append(end_line)

                lines.append("&def_nml\n")
                max_key_length = max(len(key) for key in self.ref_sources['def_nml'].keys())
                for key in list(self.ref_sources['def_nml'].keys()):
                    lines.append(f"    {key:<{max_key_length}} = {self.ref_sources['def_nml'][f'{key}']}\n")
                lines.append(end_line)
                for line in lines:
                    f1.write(line)
            time.sleep(0.5)

    def step2_mange_sources(self):
        if 'step2_remove' not in st.session_state:
            st.session_state['step2_remove'] = False
        Ref_lib = self.nl.read_namelist('./GUI/Namelist_lib/Reference_lib.nml')

        def find_source(item, source):
            found_source = False
            for key, values in Ref_lib['general'].items():
                if isinstance(values, str): values = [values]
                if key != item and source in values:
                    return False
            if not found_source:
                return True

        def get_cases(Ref_lib):
            case_item = {}
            for item in self.selected_items:
                case_item[item] = {}
                for source in Ref_lib['general'][item]:
                    case_item[item][source] = False

            st.subheader("Reference sources ....", divider=True)
            for item in case_item.keys():
                st.write(f"##### :blue[{item}]")
                cols = itertools.cycle(st.columns(3))
                for source in case_item[item].keys():
                    col = next(cols)
                    case_item[item][source] = col.checkbox(source, key=f'{item}_{source}',
                                                           value=case_item[item][source])
                    if case_item[item][source]:
                        single_source = find_source(item, source)
                        if not single_source:
                            st.warning(f"{item} {source} exist in other variables make sure you want to delete!")
            if len([f"{item}, {key}" for item in case_item.keys() for key, value in case_item[item].items() if value]) > 0:
                sources = {}
                for item in case_item.keys():
                    if any(case_item[item].values()):
                        sources[item] = {}
                    for key, value in case_item[item].items():
                        if value:
                            sources[item][key] = True
                return True, sources
            else:
                return False, {}

        cases, sources = get_cases(Ref_lib)

        def remove(sources):
            with st.spinner('Making namelist... Please wait.'):
                for item in sources:
                    for source in sources[item]:
                        remove_nml = find_source(item, source)
                        if remove_nml and source in Ref_lib['def_nml']:
                            if os.path.exists(Ref_lib['def_nml'][source]):
                                data = Ref_lib['def_nml'].pop(source)
                                try:
                                    os.remove(data)
                                    st.code(f"Remove file: {data}")
                                except:
                                    st.warning(f"{source} already removed or file doesn't exist!")
                            else:
                                st.warning(f"{source} already removed or file doesn't exist!")

                with open('./GUI/Namelist_lib/Reference_lib.nml', 'w') as f1:
                    lines = []
                    end_line = "/\n\n\n"
                    lines.append("&general\n")

                    max_key_length = max(len(key) for key in Ref_lib['general'].keys())
                    for key in list(Ref_lib['general'].keys()):
                        values = Ref_lib['general'][f'{key}']
                        if isinstance(values, str): values = [values]
                        values = [value for value in values if key not in sources or value not in sources[key]]
                        lines.append(f"    {key:<{max_key_length}} = {', '.join(values)}\n")
                    lines.append(end_line)

                    lines.append("&def_nml\n")
                    max_key_length = max(len(key) for key in Ref_lib['def_nml'].keys())
                    for key in list(Ref_lib['def_nml'].keys()):
                        if key not in sources or key not in sources.values():
                            lines.append(f"    {key:<{max_key_length}} = {Ref_lib['def_nml'][f'{key}']}\n")
                    lines.append(end_line)
                    for line in lines:
                        f1.write(line)
                time.sleep(0.8)

        def define_remove():
            st.session_state.step2_remove = True

        disable = False
        if not cases:
            disable = True

        remove_contain = st.container()

        if st.button('Remove cases', on_click=define_remove, help='Press to remove cases', disabled=disable):
            with remove_contain:
                remove(sources)
            Ref_lib = self.nl.read_namelist('./GUI/Namelist_lib/Reference_lib.nml')

        st.divider()

        def define_back(remove_contain, warn):
            if not disable:
                if not st.session_state.step2_remove:
                    with remove_contain:
                        remove(sources)
                st.session_state.step2_remove = False
            st.session_state.step2_mange_sources = False

        st.button('Finish and back to select page', on_click=define_back, args=(remove_contain, 'remove_contain'),
                  help='Finish add and go back to select page')


class make_reference(mange_reference):
    def __init__(self, initial):
        self.author = "Qingchen Xu/xuqingchen23@163.com"
        self.initial = initial

        self.nl = NamelistReader()
        self.ref_sources = self.nl.read_namelist('./GUI/Namelist_lib/Reference_lib.nml')
        self.path_finder = FindPath()
        self.base_path = Path(st.session_state.openbench_path)

        self.evaluation_items = st.session_state.evaluation_items
        self.selected_items = [k for k, v in self.evaluation_items.items() if v]
        self.tittles = [k.replace('_', ' ') for k, v in self.evaluation_items.items() if v]
        st.session_state.selected_items = self.selected_items
        st.session_state.tittles = self.tittles

        self.ref = self.initial.ref()
        self.classification = self.initial.classification()
        self.lib_path = os.path.join(st.session_state.namelist_path, 'Ref_variables_defination')

    def step2_set(self):
        if 'ref_change' not in st.session_state:
            st.session_state.ref_change = {'general': False}

        st.subheader(f'Select your Reference source', divider=True)

        if st.session_state.ref_data['general']:
            ref_general = st.session_state.ref_data['general']
        else:
            ref_general = self.ref['general']

        def ref_data_change(key, editor_key):
            ref_general[key] = st.session_state[editor_key]
            st.session_state.ref_change['general'] = True

        for selected_item in self.selected_items:
            item = f"{selected_item}_ref_source"
            if item not in ref_general:
                ref_general[item] = []
            if isinstance(ref_general[item], str): ref_general[item] = [ref_general[item]]

            label_text = f"<span style='font-size: 20px;'>{selected_item.replace('_', ' ')} reference cases ....</span>"
            st.markdown(f":blue[{label_text}]", unsafe_allow_html=True)
            if len(self.ref_sources['general'][selected_item]) == 0:
                st.warning(f"Sorry we didn't offer reference data for {selected_item.replace('_', ' ')}, please upload!")

            st.multiselect("Reference offered",
                           [value for value in self.ref_sources['general'][selected_item]],
                           default=[value for value in ref_general[item] if value in self.ref_sources['general'][selected_item]],
                           key=f"{item}_multi",
                           on_change=ref_data_change,
                           args=(item, f"{item}_multi"),
                           placeholder="Choose an option",
                           label_visibility="collapsed")

        st.session_state.step2_set_check = self.__step2_setcheck(ref_general)

        sources = list(set([value for key in self.selected_items for value in ref_general[f"{key}_ref_source"] if value]))
        st.session_state.ref_data['def_nml'] = {}
        for source in sources:
            st.session_state.ref_data['def_nml'][source] = self.ref_sources['def_nml'][source]
            if source not in st.session_state.ref_change:
                st.session_state.ref_change[source] = False

        for source in st.session_state.ref_data.keys():
            if source not in sources + ['general', 'def_nml']:
                del st.session_state.ref_data[source]

        formatted_keys = " \n".join(
            f'{key.replace("_", " ")}: {", ".join(value for value in ref_general[f"{key}_ref_source"] if value)}' for
            key in
            self.selected_items)
        sourced_key = " \n".join(f"{source}: {self.ref_sources['def_nml'][source]}" for source in sources)
        st.code(f'''{formatted_keys}\n\n{sourced_key}''', language="shell", line_numbers=True, wrap_lines=True)

        col1, col, col3 = st.columns(3)

        def define_new_refnml():
            st.session_state.step2_make_newnamelist = True

        col1.button('Add new reference namelist', on_click=define_new_refnml)

        def define_clear_sources():
            st.session_state.step2_mange_sources = True

        col3.button('Manage Reference sources', on_click=define_clear_sources)

        st.session_state.ref_data['general'] = ref_general

        def define_step1():
            st.session_state.step2_set = False

        def define_step2():
            st.session_state.step2_make = True

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Evalution page')

        with col4:
            st.button('Next step :soon: ', on_click=define_step2, help='Go to Making page')

    def __step2_setcheck(self, general_ref):
        check_state = 0

        for selected_item in self.selected_items:
            key = f'{selected_item}_ref_source'
            if (general_ref[key] == []) or (general_ref[key] is None) or (len(general_ref[key]) == 0):
                st.error(f'Please choose at least one source data in {key.replace("_", " ")}!', icon="⚠")
                check_state += 1
            if selected_item not in st.session_state.step2_errorlist:
                st.session_state.step2_errorlist[selected_item] = []

            if check_state > 0:
                st.session_state.step2_errorlist[selected_item].append(1)
                st.session_state.step2_errorlist[selected_item] = list(np.unique(st.session_state.step2_errorlist[selected_item]))
                return False
            if check_state == 0:
                if (selected_item in st.session_state.step2_errorlist) & (
                        1 in st.session_state.step2_errorlist[selected_item]):
                    st.session_state.step2_errorlist[selected_item] = list(
                        filter(lambda x: x != 1, st.session_state.step2_errorlist[selected_item]))
                    st.session_state.step2_errorlist[selected_item] = list(
                        np.unique(st.session_state.step2_errorlist[selected_item]))
                return True

    def step2_make(self):
        if 'step2_make_check' not in st.session_state:
            st.session_state.step2_make_check = False

        ref_general = st.session_state.ref_data['general']
        def_nml = st.session_state.ref_data['def_nml']

        if ref_general and def_nml:
            st.session_state.step2_check = []
            for (source, path), tab in zip(def_nml.items(), st.tabs(def_nml.keys())):
                try:
                    if source not in st.session_state.ref_data:
                        st.session_state.ref_data[source] = self.nl.read_namelist(path)
                    tab.subheader(f':blue[{source} Reference checking ....]', divider=True)
                    with tab:
                        self.__step2_make_ref_info(source, st.session_state.ref_data[source], ref_general)
                except Exception as e:
                    st.error(f'Error: {e}')
            if all(st.session_state.step2_check):
                st.session_state.step2_make_check = True
            else:
                st.session_state.step2_make_check = False
        else:
            st.error('Please select your Reference first!')
            st.session_state.step2_make_check = False

        def define_step1():
            st.session_state.step2_make = False

        def define_step2():
            st.session_state.step2_tomake_nml = True

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Reference set page',
                      use_container_width=True)
        with col4:
            st.button('Next step :soon: ', on_click=define_step2, help='Go to Reference nml page',
                      use_container_width=True)

    def __step2_make_ref_info(self, source, source_lib, ref_general):

        def ref_editor_change(key, editor_key, source):
            source_lib[key][editor_key] = st.session_state[f"{source}_{key}_{editor_key}"]
            st.session_state.ref_change[source] = True

        import itertools
        with st.container(height=None, border=True):
            key = 'general'
            if 'root_dir' not in source_lib[key]: source_lib[key][f"root_dir"] = './data/'
            if not source_lib[key][f"root_dir"]: source_lib[key][f"root_dir"] = './data/'
            find_path = self.path_finder.find_path(source_lib['general'][f"root_dir"], f"{source}_general_root_dir",
                                                   ['ref_change', source])
            st.code(f"Dictionary: {find_path}", language='shell', wrap_lines=True)
            source_lib[key][f"root_dir"] = find_path

            if source_lib[key]['data_type'] == 'stn':
                if 'Streamflow' not in source_lib.keys():
                    if 'fulllist' not in source_lib[key]: source_lib[key][f"fulllist"] = None
                    if not source_lib[key][f"fulllist"]: source_lib[key][f"fulllist"] = None
                    source_lib[key][f"fulllist"] = self.path_finder.get_file(source_lib[key][f"fulllist"],
                                                                             f"{source}_{key}_fulllist",
                                                                             'csv', ['ref_change', source])
                    st.code(f"Set Fulllist File: {source_lib[key][f'fulllist']}", language='shell', wrap_lines=True)

            cols = itertools.cycle(st.columns(2))
            for key, values in source_lib.items():
                if key != 'general' and key in self.selected_items:
                    if source in ref_general[f'{key}_ref_source']:
                        if key == 'Streamflow':
                            for info in ['max_uparea', 'min_uparea']:
                                col = next(cols)
                                source_lib[key][info] = col.number_input(info.title().replace("_", " "),
                                                                         value=source_lib[key][info],
                                                                         key=f"{key}_{info}",
                                                                         on_change=ref_editor_change,
                                                                         args=(key, info, source),
                                                                         placeholder=f"Set your Reference {info.title().replace("_", " ")}...")
                        if 'sub_dir' in source_lib[key].keys():
                            col = next(cols)
                            with col:
                                st.write(f'Set {key.replace("_", " ")} Sub-Data Dictionary:')
                                if not source_lib[key][f"sub_dir"]: source_lib[key][f"sub_dir"] = ''
                                source_lib[key][f"sub_dir"] = self.path_finder.find_subdirectories(source_lib[key][f"sub_dir"],
                                                                                                   f"{source}_general_root_dir",
                                                                                                   f"{source}_{key}_sub_dir",
                                                                                                   ['ref_change', source])
                                st.code(f"Sub-Dir: {source_lib[key][f'sub_dir']}", language='shell', wrap_lines=True)

            st.session_state.step2_check.append(self.__step2_makecheck(source_lib, source))

    def __step2_makecheck(self, source_lib, source):
        error_state = 0
        warning_state = 0

        general_list = ["root_dir", "timezone", "data_type", "data_groupby", "tim_res"]
        grid_list = ["grid_res", "syear", "eyear"]
        stn_list = ["fulllist"]
        info_list = ['sub_dir', 'varname', 'varunit', 'prefix', 'suffix', 'syear', 'eyear']
        for key in general_list:
            if isinstance(source_lib['general'][key], str):
                if len(source_lib['general'][key]) < 1:
                    error_state += 1
                else:
                    warning_state += 1

        if source_lib['general']["data_type"] == 'grid':
            if "grid_res" in source_lib['general'].keys():
                if isinstance(source_lib['general']["grid_res"], float) | isinstance(source_lib['general']["grid_res"],
                                                                                     int):
                    if source_lib['general']["grid_res"] <= 0:
                        st.error(
                            f"general: Geo Resolution should be larger than zero when data_type is 'geo', please check.",
                            icon="⚠")
                        error_state += 1
            if "syear" in source_lib['general'].keys() and "eyear" in source_lib['general'].keys():
                if isinstance(source_lib['general']["syear"], int) and isinstance(source_lib['general']["eyear"], int):
                    if source_lib['general']["syear"] > source_lib['general']["eyear"]:
                        st.error(f'general: End year should be larger than Start year, please check.',
                                 icon="⚠")
                        error_state += 1
        else:
            if 'Streamflow' not in source_lib.keys():
                if not source_lib['general']["fulllist"]:
                    error_state += 1
            else:
                if isinstance(source_lib['general']["fulllist"], str):
                    if len(source_lib['general']["fulllist"]) < 1:
                        warning_state += 1

        for var in source_lib.keys():
            if var != 'general':
                for key in source_lib[var].keys():
                    if key in ['varname', 'varunit']:  #
                        if var in self.selected_items and source in st.session_state.ref_data['general'][f'{var}_ref_source']:
                            if len(source_lib[var][key]) < 1:
                                st.error(f'{var}: {key} should be a string longer than one, please check.', icon="⚠")
                                error_state += 1
                    # elif key == 'sub_dir':
                    #     if source_lib[var][key] is None:
                    #         st.warning(f'{var}: {key} should be a string, please check.',
                    #                    icon="⚠")
                    #     else:
                    #         if len(source_lib[var][key]) < 1:
                    #             st.warning(f'{var}: {key} should be a string longer than one, please check.',
                    #                        icon="⚠")
                    elif key in ['prefix', 'suffix']:
                        if len(source_lib[var]['prefix']) < 1 and len(source_lib[var]['suffix']) < 1:
                            warning_state += 1
                    elif key in ['syear', 'eyear']:
                        if not isinstance(source_lib[var][key], int):
                            warning_state += 1
                        if source_lib[var]["syear"] > source_lib[var]["eyear"]:
                            error_state += 1

        if warning_state > 0:
            st.warning(f"Some mistake in source, please check your file!", icon="⚠")

        if source not in st.session_state.step2_errorlist:
            st.session_state.step2_errorlist[source] = []
        if error_state > 0:
            st.session_state.step2_errorlist[source].append(2)
            st.session_state.step2_errorlist[source] = list(np.unique(st.session_state.step2_errorlist[source]))
            return False
        if error_state == 0:
            if (source in st.session_state.step2_errorlist) & (2 in st.session_state.step2_errorlist[source]):
                st.session_state.step2_errorlist[source] = list(
                    filter(lambda x: x != 2, st.session_state.step2_errorlist[source]))
                st.session_state.step2_errorlist[source] = list(np.unique(st.session_state.step2_errorlist[source]))
            return True

    def step2_ref_nml(self):
        step2_disable = False
        if st.session_state.step2_set_check & st.session_state.step2_make_check:
            for source, path in st.session_state.ref_data['def_nml'].items():
                source_lib = st.session_state.ref_data[source]
                st.subheader(source, divider=True)
                path_info = f'Root Dictionary: {source_lib["general"][f"root_dir"]}'
                key = 'general'
                if source_lib[key]['data_type'] == 'stn':
                    path_info = path_info + f'\nFulllist File: {source_lib["general"][f"fulllist"]}'

                for key, values in source_lib.items():
                    if key != 'general' and key in self.selected_items:
                        if source in st.session_state.ref_data['general'][f'{key}_ref_source']:
                            if 'sub_dir' in source_lib[key].keys():
                                path_info = path_info + f'\n{key.replace("_", " ")} Sub-Data Dictionary: {source_lib[key][f"sub_dir"]}'
                st.code(f'''{path_info}''', language='shell', line_numbers=True, wrap_lines=True)
            st.session_state.step2_ref_check = True
            st.session_state.step2_ref_nml = False
        else:
            step2_disable = True
            if not st.session_state.step2_set_check:
                formatted_keys = ", ".join(
                    key.replace('_', ' ') for key, value in st.session_state.step2_errorlist.items() if 1 in value)
                st.error(
                    f'There exist error in set page, please check {formatted_keys} first! Set your reference data.',
                    icon="🚨")
            if not st.session_state.step2_make_check:
                formatted_keys = ", ".join(
                    key.replace('_', ' ') for key, value in st.session_state.step2_errorlist.items() if 2 in value)
                st.error(f'There exist error in Making page, please check {formatted_keys} first!', icon="🚨")
            st.session_state.step2_ref_nml = False
            st.session_state.step2_ref_check = False

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
                    max_key_length = max(len(key) for key in variables.keys())
                    if section == 'general':
                        continue  # 'general' 已经处理过了
                    f.write(f'&{section}\n')
                    for key, value in variables.items():
                        f.write(f'  {key:<{max_key_length}} = {value}\n')
                    f.write('/\n\n')
            del f

        def make():
            for key, value in st.session_state.ref_change.items():
                if key == 'general' and value:
                    if st.session_state.step2_ref_check & (not st.session_state.step2_ref_nml):
                        st.session_state.step2_ref_nml = self.__step2_make_ref_namelist(
                            st.session_state.generals['reference_nml'], self.selected_items, st.session_state.ref_data)
                    if st.session_state.step2_ref_nml:
                        st.success("😉 Make file successfully!!!")
                        st.session_state.ref_change[key] = False
                elif key != 'general' and value:
                    file = st.session_state.ref_data['def_nml'][key]
                    source_lib = st.session_state.ref_data[key]
                    write_nml(source_lib, file)
                    st.session_state.ref_change[key] = False

        def define_step1():
            st.session_state.step2_tomake_nml = False

        def define_step2(make_contain, smake):
            if not st.session_state.step2_ref_check:
                st.session_state.step3_set = False
                st.session_state.step2 = False
            else:
                with make_contain:
                    if any(v for v in st.session_state.ref_change.values()):
                        make()
                st.session_state.step3_set = True
                st.session_state.step2 = True

        make_contain = st.container()

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Reference Making page')
        with col4:
            st.button('Next step :soon: ', on_click=define_step2, args=(make_contain, 'make'), disabled=step2_disable,
                      help='Go to Simulation page')

    def __step2_make_ref_namelist(self, file_path, selected_items, ref_data):
        general = ref_data['general']
        def_nml = ref_data['def_nml']

        with st.spinner('Making namelist... Please wait.'):
            if st.session_state.step2_ref_check:
                st.write("Making namelist...")
                with open(file_path, 'w') as f:
                    lines = []
                    end_line = "/\n\n\n"

                    lines.append("&general\n")
                    max_key_length = max(len(f"{key}") for key in general.keys())
                    for item in selected_items:
                        key = f"{item}_ref_source"
                        lines.append(f"    {key:<{max_key_length}} = {','.join(general[f'{item}_ref_source'])}\n")
                    lines.append(end_line)

                    lines.append("&def_nml\n")
                    max_key_length = max(len(key) for key in def_nml.keys())
                    for key, value in def_nml.items():
                        lines.append(f"    {key:<{max_key_length}} = {value}\n")
                    lines.append(end_line)
                    for line in lines:
                        f.write(line)
                    time.sleep(2)

                    return True
