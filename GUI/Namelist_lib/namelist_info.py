# -*- coding: utf-8 -*-
import numpy as np
import sys


class initial_setting():
    def __init__(self):
        self.author = "Qingchen Xu/xuqingchen0@gmail.com"
        self.info = "Store initialization information"

    def main(self):
        info = {
            'general': {},
            'evaluation_items': {
            },
            'metrics': {
            },
            'scores': {
            },
            'comparisons': {},
            'statistics': {}
        }
        return info

    def sim(self):
        info = {
            'general': {
            },
            'def_nml': {},
        }
        return info

    def ref(self):
        info = {
            'general': {},
            'def_nml': {},
        }
        return info

    def stat(self):
        info = {
            'general': {}
        }
        return info

    # -------------------------------
    def generals(self):
        General = {
            'basename': '',
            'basedir': '/',
            'compare_tim_res': 'month',
            'compare_tzone': 0.0,
            'compare_grid_res': 0.25,
            'syear': 2002,
            'eyear': 2003,
            'min_year': 1.0,
            'max_lat': 90.0,
            'min_lat': -90.0,
            'max_lon': 180.0,
            'min_lon': -180.0,
            'reference_nml': '',
            'simulation_nml': '',
            'statistics_nml': '',
            'figure_nml': '',
            'num_cores': 8,
            'evaluation': True,
            'comparison': False,
            'statistics': False,
            'debug_mode': True,
            'weight': 'None',  # area, mass  # weight for metrics and scores
            'IGBP_groupby': True,  # True: show metics and scores grouped by IGBP
            'PFT_groupby': True,  # True: show metics and scores grouped by PFT
            'unified_mask': True,  # True: mask the observation data with all simulation datasets to ensure consistent coverage
        }
        return General

    def evaluation_items(self):
        option = {
            'Gross_Primary_Productivity': False,
            'Ecosystem_Respiration': False,
            'Net_Ecosystem_Exchange': False,

            'Leaf_Area_Index': False,
            'Biomass': False,
            'Burned_Area': False,
            'Soil_Carbon': False,
            'Nitrogen_Fixation': False,
            'Methane': False,
            'Veg_Cover_In_Fraction': False,
            'Leaf_Greenness': False,
            # **************************************************************

            # *******************      Hydrology Cycle      ****************
            ###surface####
            'Evapotranspiration': False,
            'Canopy_Transpiration': False,
            'Canopy_Interception': False,
            'Ground_Evaporation': False,
            'Water_Evaporation': False,
            'Soil_Evaporation': False,
            'Total_Runoff': False,
            'Terrestrial_Water_Storage_Change': False,

            ###Snow####
            'Snow_Water_Equivalent': False,
            'Surface_Snow_Cover_In_Fraction': False,
            'Snow_Depth': False,
            'Permafrost': False,

            ###Soil####
            'Surface_Soil_Moisture': False,
            'Root_Zone_Soil_Moisture': False,

            ###  Groundwater ####
            'Water_Table_Depth': False,
            'Water_Storage_In_Aquifer': False,
            'Depth_Of_Surface_Water': False,
            'Groundwater_Recharge_Rate': False,
            # **************************************************************

            # *******************  Radiation and Energy Cycle  *************
            'Net_Radiation': False,
            'Latent_Heat': False,
            'Sensible_Heat': False,
            'Ground_Heat': False,
            'Albedo': False,
            'Surface_Upward_SW_Radiation': False,
            'Surface_Upward_LW_Radiation': False,
            'Surface_Net_SW_Radiation': False,
            'Surface_Net_LW_Radiation': False,
            'Surface_Soil_Temperature': False,
            'Root_Zone_Soil_Temperature': False,
            # ****************************************************************

            # *******************         Forcings      **********************
            'Diurnal_Temperature_Range': False,
            'Diurnal_Max_Temperature': False,
            'Diurnal_Min_Temperature': False,
            'Surface_Downward_SW_Radiation': False,
            'Surface_Downward_LW_Radiation': False,
            'Surface_Relative_Humidity': False,
            'Surface_Specific_Humidity': False,
            'Precipitation': False,
            'Surface_Air_Temperature': False,
            # ****************************************************************

            # *******************      Human Activity    **********************
            # urban
            'Urban_Anthropogenic_Heat_Flux': False,
            'Urban_Albedo': False,
            'Urban_Surface_Temperature': False,
            'Urban_Air_Temperature_Max': False,
            'Urban_Air_Temperature_Min': False,
            'Urban_Latent_Heat_Flux': False,

            # Crop
            'Crop_Yield_Rice': False,
            'Crop_Yield_Corn': False,
            'Crop_Yield_Wheat': False,
            'Crop_Yield_Maize': False,
            'Crop_Yield_Soybean': False,
            'Crop_Heading_DOY_Corn': False,  # under develop
            'Crop_Heading_DOY_Wheat': False,  # under develop
            'Crop_Maturity_DOY_Corn': False,  # under develop
            'Crop_Maturity_DOY_Wheat': False,  # under develop
            'Crop_V3_DOY_Corn': False,  # under develop
            'Crop_Emergence_DOY_Wheat': False,  # under develop

            'Total_Irrigation_Amount': False,

            ###Dam###
            'Dam_Inflow': False,
            'Dam_Outflow': False,
            'Dam_Water_Storage': False,
            'Dam_Water_Elevation': False,

            ###Lake####
            'Lake_Temperature': False,
            'Lake_Ice_Fraction_Cover': False,
            'Lake_Water_Level': False,
            'Lake_Water_Area': False,
            'Lake_Water_Volume': False,

            ###River####
            'Streamflow': False,
            'Inundation_Fraction': False,
            'Inundation_Area': False,
            'River_Water_Level': False,
        }
        return option

    def metrics(self):
        metric = {
            'percent_bias': False,  # Percent Bias
            'absolute_percent_bias': False,  # Absolute Percent Bias
            'bias': False,  # Bias
            'mean_absolute_error': False,  # Mean Absolute Error

            'RMSE': False,  # Root Mean Squared Error
            'MSE': False,  # Mean Squared Error
            'ubRMSE': False,  # Unbiased Root Mean Squared Error
            'CRMSD': False,  # Centered Root Mean Square Difference
            'nrmse': False,  # Normalized Root Mean Square Error
            'L': False,  # Likelihood
            'correlation': False,  # correlation coefficient
            'correlation_R2': False,  # correlation coefficient R2
            'NSE': False,  # Nash Sutcliffe efficiency coefficient
            'LNSE': False,  # natural logarithm of NSE coefficient
            'KGE': False,  # Kling-Gupta Efficiency
            'KGESS': False,  # Normalized Kling-Gupta Efficiency
            'kappa_coeff': False,  # Kappa coefficient
            'rv': False,  # Relative variability (amplitude ratio)
            'ubNSE': False,  # Unbiased Nash Sutcliffe efficiency coefficient
            'ubKGE': False,  # Unbiased Kling-Gupta Efficiency
            'ubcorrelation': False,  # Unbiased correlation
            'ubcorrelation_R2': False,  # correlation coefficient R2
            'pc_max': False,  # the bias of the maximum value
            'pc_min': False,  # the bias of the minimum value
            'pc_ampli': False,  # the bias of the amplitude value
            'rSD': False,  # Ratio of standard deviations
            'PBIAS_HF': False,  # Percent bias of flows ≥ Q98 (Yilmaz et al., 2008)
            'PBIAS_LF': False,  # Percent bias of flows ≤ Q30(Yilmaz et al., 2008)
            'SMPI': False,  # https://docs.esmvaltool.org/en/latest/recipes/recipe_smpi.html
            'ggof': False,  # Graphical Goodness of Fit
            'gof': False,  # Numerical Goodness-of-fit measures
            'KGEkm': False,  # Kling-Gupta Efficiency with knowable-moments
            'KGElf': False,  # Kling-Gupta Efficiency for low values
            'KGEnp': False,  # Non-parametric version of the Kling-Gupta Efficiency
            'md': False,  # Modified Index of Agreement
            'mNSE': False,  # Modified Nash-Sutcliffe efficiency
            'pbiasfdc': False,  # Percent Bias in the Slope of the Midsegment of the Flow Duration Curve
            'pfactor': False,  # the percent of observations that are within the given uncertainty bounds.
            'rd': False,  # Relative Index of Agreement
            'rfactor': False,
            # the average width of the given uncertainty bounds divided by the standard deviation of the observations.
            'rNSE': False,  # Relative Nash-Sutcliffe efficiency
            'rSpearman': False,  # Spearman’s rank correlation coefficient
            'rsr': False,  # Ratio of RMSE to the standard deviation of the observations
            'sKGE': False,  # Split Kling-Gupta Efficiency
            'ssq': False,  # Sum of the Squared Residuals
            'valindex': False,  # Valid Indexes
            've': False,  # Volumetric Efficiency
            'wNSE': False,  # Weighted Nash-Sutcliffe efficiency
            'wsNSE': False,  # Weighted seasonal Nash-Sutcliffe Efficiency
            'index_agreement': False,  # Index of agreement
        }
        return metric

    def scores(self):
        scores = {
            'nBiasScore': False,  # Bias Score from ILAMB
            'nRMSEScore': False,  # RMSE Score from ILAMB
            'nPhaseScore': False,  # Phase Score from ILAMB
            'nIavScore': False,  # Interannual Variability Score from ILAMB
            'nSpatialScore': False,  # Spatial distribution score
            'Overall_Score': True,  # overall score from ILAMB
            'The_Ideal_Point_score': False,
        }
        return scores

    def comparisons(self):
        comparisons = {
            'HeatMap': False,
            'Taylor_Diagram': False,
            'Target_Diagram': False,
            'Kernel_Density_Estimate': False,
            'Whisker_Plot': False,  # not ready
            'Parallel_Coordinates': False,
            'Portrait_Plot_seasonal': False,
            'Single_Model_Performance_Index': False,
            'Relative_Score': False,
            'Ridgeline_Plot': False,
            'Diff_Plot': False,
            'Mean': False,
            'Median': False,
            'Min': False,
            'Max': False,
            'Sum': False,
            'Mann_Kendall_Trend_Test': False,
            'Correlation': False,
            'Standard_Deviation': False,
            'Functional_Response': False,
        }
        return comparisons

    def statistics(self):
        statistics = {
            'Z_Score': False,
            'Hellinger_Distance': False,
            'Partial_Least_Squares_Regression': False,
            'Three_Cornered_Hat': False,
            'ANOVA': False
        }
        return statistics

    # -------------------------------
    def classification(self):
        classifications = {
            "Ecosystem and Carbon Cycle":
                {'Gross_Primary_Productivity',
                 'Ecosystem_Respiration',
                 'Net_Ecosystem_Exchange',

                 'Leaf_Area_Index',
                 'Biomass',
                 'Burned_Area',
                 'Soil_Carbon',
                 'Nitrogen_Fixation',
                 'Methane',
                 'Veg_Cover_In_Fraction',
                 'Leaf_Greenness',
                 },
            "Hydrology Cycle": {
                'Evapotranspiration',
                'Canopy_Transpiration',
                'Canopy_Interception',
                'Ground_Evaporation',
                'Water_Evaporation',
                'Soil_Evaporation',
                'Total_Runoff',
                'Terrestrial_Water_Storage_Change',

                ###Snow####
                'Snow_Water_Equivalent',
                'Surface_Snow_Cover_In_Fraction',
                'Snow_Depth',
                'Permafrost',

                ###Soil####
                'Surface_Soil_Moisture',
                'Root_Zone_Soil_Moisture',

                ###  Groundwater ####
                'Water_Table_Depth',
                'Water_Storage_In_Aquifer',
                'Depth_Of_Surface_Water',
                'Groundwater_Recharge_Rate',
            },
            "Radiation and Energy Cycle": {
                'Net_Radiation',
                'Latent_Heat',
                'Sensible_Heat',
                'Ground_Heat',
                'Albedo',
                'Surface_Upward_SW_Radiation',
                'Surface_Upward_LW_Radiation',
                'Surface_Net_SW_Radiation',
                'Surface_Net_LW_Radiation',
                'Surface_Soil_Temperature',
                'Root_Zone_Soil_Temperature',
            },

            "Forcings": {
                'Diurnal_Temperature_Range',
                'Diurnal_Max_Temperature',
                'Diurnal_Min_Temperature',
                'Surface_Downward_SW_Radiation',
                'Surface_Downward_LW_Radiation',
                'Surface_Relative_Humidity',
                'Surface_Specific_Humidity',
                'Precipitation',
                'Surface_Air_Temperature', },
            "Human Activity": {
                # *******************      Human Activity    **********************
                # urban
                'Urban_Anthropogenic_Heat_Flux',
                'Urban_Albedo',
                'Urban_Surface_Temperature',
                'Urban_Air_Temperature_Max',
                'Urban_Air_Temperature_Min',
                'Urban_Latent_Heat_Flux',

                # Crop
                'Crop_Yield_Rice',
                'Crop_Yield_Corn',
                'Crop_Yield_Wheat',
                'Crop_Yield_Maize',
                'Crop_Yield_Soybean',
                'Crop_Heading_DOY_Corn',  # under develop
                'Crop_Heading_DOY_Wheat',  # under develop
                'Crop_Maturity_DOY_Corn',  # under develop
                'Crop_Maturity_DOY_Wheat',  # under develop
                'Crop_V3_DOY_Corn',  # under develop
                'Crop_Emergence_DOY_Wheat',  # under develop

                'Total_Irrigation_Amount',

                ###Dam###
                'Dam_Inflow',
                'Dam_Outflow',
                'Dam_Water_Storage',
                'Dam_Water_Elevation',

                ###Lake####
                'Lake_Temperature',
                'Lake_Ice_Fraction_Cover',
                'Lake_Water_Level',
                'Lake_Water_Area',
                'Lake_Water_Volume',

                ###River####
                'Streamflow',
                'Inundation_Fraction',
                'Inundation_Area',
                'River_Water_Level', }
        }
        return classifications

        # info = {
        #     'general': {
        #         'Evapotranspiration_sim_source': 'single_point',
        #         'Latent_Heat_sim_source': 'test1',
        #         'Streamflow_sim_source': 'test2'
        #     },
        #     'Evapotranspiration': {
        #         '_model': 'CoLM',
        #         '_timezone': 'Local',
        #         '_data_type': 'stn',
        #         '_data_groupby': 'Month',
        #         '_dir': '/stu01/caiyt18/CoLM_medlyn/cases/',
        #         '_varname': 'f_fevpa',
        #         '_fulllist': '/tera04/zhwei/colm/new/sim-test.csv',
        #         '_tim_res': 'Hour',
        #         '_geo_res': '',
        #         '_suffix': '',
        #         '_prefix': '',
        #         '_syear': '',
        #         '_eyear': ''
        #     },
        #     'Latent_Heat': {
        #         '_model': 'CoLM',
        #         '_timezone': 0,
        #         '_data_type': 'geo',
        #         '_data_groupby': 'Month',
        #         '_dir': '/tera04/zhwei/colm/cases/global_era5_igbp_unstructure_0.5Dc_r/history',
        #         '_varname': 'f_lfevpa',
        #         '_tim_res': 'Day',
        #         '_geo_res': 0.5,
        #         '_suffix': 'global_era5_igbp_unstructure_0.5Dc_r_hist_',
        #         '_prefix': '',
        #         '_syear': 2001,
        #         '_eyear': 2020
        #     },
        #     'Streamflow': {
        #         '_model': 'CoLM',
        #         '_timezone': 0,
        #         '_data_type': 'geo',
        #         '_data_groupby': 'Year',
        #         '_dir': '/tera04/zhwei/cama/CaMa-Flood_v411-20231010/out/ERA5LAND-grided-bc-15min-filled',
        #         '_varname': 'outflw',
        #         '_tim_res': 'Day',
        #         '_geo_res': 0.25,
        #         '_suffix': 'o_outflw',
        #         '_prefix': '',
        #         '_syear': 2001,
        #         '_eyear': 2018,
        #         'test2_model': 'CoLM',
        #         'test2timezone': 0,
        #         'test2_data_type': 'geo',
        #         'test2_data_groupby': 'Year',
        #         'test2_dir': '/tera05/zhangsp/cases/gridbased_crujra_igbp_glb/history0',
        #         'test2_varname': 'outflw',
        #         'test2_tim_res': 'Month',
        #         'test2_geo_res': 0.25,
        #         'test2_suffix': 'gridbased_crujra_igbp_glb_hist_cama_',
        #         'test2_prefix': '',
        #         'test2_syear': 2002,
        #         'test2_eyear': 2018
        #     }
        # }
    # -------------------------------
    def stat_list(self):
        statistics = {
            'Mann_Kendall_Trend_Test': {
                'general': ['timezone', 'data_type', 'data_groupby', 'dir', 'fulllist', 'varname', 'tim_res', 'grid_res',
                            'suffix',
                            'prefix', 'syear', 'eyear'], 'other': ['significance_level']},
            'Correlation': {
                'general': ['timezone', 'data_type', 'data_groupby', 'dir', 'fulllist', 'varname', 'tim_res', 'grid_res',
                            'suffix',
                            'prefix', 'syear', 'eyear']},
            'Standard_Deviation': {
                'general': ['timezone', 'data_type', 'data_groupby', 'dir', 'fulllist', 'varname', 'tim_res', 'grid_res',
                            'suffix',
                            'prefix', 'syear', 'eyear']},
            'Z_Score': {
                'general': ['timezone', 'data_type', 'data_groupby', 'dir', 'fulllist', 'varname', 'tim_res', 'grid_res',
                            'suffix',
                            'prefix', 'syear', 'eyear']},
            'Functional_Response': {
                'general': ['timezone', 'data_type', 'data_groupby', 'dir', 'fulllist', 'varname', 'tim_res', 'grid_res',
                            'suffix',
                            'prefix', 'syear', 'eyear'], 'other': ['nbins']},
            'Hellinger_Distance': {
                'general': ['timezone', 'data_type', 'data_groupby', 'dir', 'fulllist', 'varname', 'tim_res', 'grid_res',
                            'suffix',
                            'prefix', 'syear', 'eyear'], 'other': ['nbins']},
            'Partial_Least_Squares_Regression': {
                'general': ['timezone', 'data_type', 'data_groupby', 'dir', 'fulllist', 'varname', 'tim_res', 'grid_res',
                            'suffix',
                            'prefix', 'syear', 'eyear'], 'other': ['max_components', 'n_splits', 'n_jobs', ]},
            'Three_Cornered_Hat': {
                'general': ['timezone', 'data_type', 'data_groupby', 'dir', 'fulllist', 'varname', 'tim_res', 'grid_res',
                            'suffix',
                            'prefix', 'syear', 'eyear']}
        }
        return statistics

    def stat_default(self):
        default = {
            'timezone': 0.0,
            'data_type': 'grid',
            'data_groupby': 'Day',
            'dir': '',
            'varname': '',
            'fulllist': '',
            'tim_res': 'Day',
            'varunit': '',
            'grid_res': 0.5,
            'suffix': '',
            'prefix': '',
            'syear': 2001,
            'eyear': 2020
        }
        return default