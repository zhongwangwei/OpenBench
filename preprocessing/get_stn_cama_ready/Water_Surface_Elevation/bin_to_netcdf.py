#!/usr/bin/env python3
import os
import numpy as np
import netCDF4 as nc
import argparse
from pathlib import Path
import datetime
import sys
from set_name import set_name, get_region_boundaries

def convert_binary_to_netcdf(binary_file, output_file, shape, dtype, description=None, west=None, east=None, south=None, north=None, gsize=None, var_name=None, units=None, force_overwrite=False):
    """
    将二进制文件转换为标准NetCDF格式
    
    参数:
        binary_file (str): 二进制文件路径
        output_file (str): 输出NetCDF文件路径
        shape (tuple): 数组形状
        dtype (str): 数据类型
        description (str): 变量描述
        west (float): 西边界经度
        east (float): 东边界经度
        south (float): 南边界纬度
        north (float): 北边界纬度
        gsize (float): 网格大小（度）
        var_name (str): 变量名称，默认从文件名提取
        units (str): 变量单位
        force_overwrite (bool): 是否强制覆盖已存在的文件
    """
    # 检查目标文件是否已存在
    if not force_overwrite and os.path.exists(output_file):
        try:
            # 尝试打开文件检查是否是有效的NetCDF文件
            with nc.Dataset(output_file, 'r') as ds:
                if var_name is None:
                    var_name = os.path.basename(binary_file).split('.')[0]
                
                # 检查变量是否存在
                if var_name in ds.variables:
                    print(f"File {output_file} already exists and contains variable {var_name}. Skipping conversion.")
                    return True
                else:
                    print(f"File {output_file} exists but does not contain variable {var_name}. Will overwrite.")
        except:
            print(f"File {output_file} exists but is not a valid NetCDF file. Will overwrite.")
    
    try:
        # 检查输入文件是否存在
        if not os.path.exists(binary_file):
            print(f"Error: Input file {binary_file} does not exist")
            return False

        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # 处理数据类型
        if dtype == np.float32 or dtype == 'float32' or str(dtype) == "<class 'numpy.float32'>":
            read_dtype = '<f4'
            nc_dtype = 'f4'
            fill_value = -9999.0
        elif dtype == np.int32 or dtype == 'int32' or str(dtype) == "<class 'numpy.int32'>":
            read_dtype = '<i4'
            nc_dtype = 'i4'
            fill_value = -9999
        elif dtype == np.int8 or dtype == 'int8' or str(dtype) == "<class 'numpy.int8'>":
            read_dtype = '<i1'
            nc_dtype = 'i1'
            fill_value = -99  # int8范围是-128到127，所以使用-99而不是-9999
        elif dtype == np.int16 or dtype == 'int16' or str(dtype) == "<class 'numpy.int16'>":
            read_dtype = '<i2'
            nc_dtype = 'i2'
            fill_value = -999  # int16范围是-32768到32767，可以使用-999
        else:
            print(f"Warning: Unknown dtype {dtype}, trying as string")
            read_dtype = '<' + str(dtype).replace("<class 'numpy.", "").replace("'>", "")
            nc_dtype = str(dtype).replace("<class 'numpy.", "").replace("'>", "")
            fill_value = -9999.0
            
        # 读取二进制数据 - 使用little-endian格式
        try:
            with open(binary_file, 'rb') as f:
                data = np.fromfile(f, dtype=read_dtype)
                data = data.reshape(shape)
        except ValueError as e:
            print(f"Error reshaping data from {binary_file} to shape {shape}: {e}")
            print(f"Data length: {len(data)}, Required shape: {shape}, Required elements: {shape[0]*shape[1]}")
            return False
            
        # 如果没有指定变量名，从文件名提取
        if var_name is None:
            var_name = os.path.basename(binary_file).split('.')[0]
            
        # 创建NetCDF文件
        ds = nc.Dataset(output_file, 'w', format='NETCDF4')
        
        # 添加全局属性
        ds.title = f"CaMa-Flood {var_name} data"
        ds.description = description or f'Converted from {binary_file}'
        ds.history = f'Created {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        ds.source = 'CaMa-Flood binary file conversion'
        ds.Conventions = 'CF-1.8'
        ds.institution = "Local Data Processing"
        ds.references = "CaMa-Flood model"
        
        # 创建维度
        ds.createDimension('lat', shape[0])
        ds.createDimension('lon', shape[1])
        
        # 创建坐标变量
        if all(x is not None for x in [west, east, south, north, gsize]):
            # 计算经纬度值，确保不会出现NaN
            # 首先确保网格数量与维度匹配
            nlon = shape[1]
            nlat = shape[0]
            
            # 创建经度变量
            lon = ds.createVariable('lon', 'f8', ('lon',))
            lon.units = 'degrees_east'
            lon.long_name = 'longitude'
            lon.standard_name = 'longitude'
            lon.axis = 'lon'  # 使用 lon/lat
            lon.valid_min = west
            lon.valid_max = east
            
            # 使用linspace确保准确的点数，避免舍入误差
            # 从west+gsize/2开始到east，确保不会超出范围
            lon_values = np.linspace(west + gsize/2, east - gsize/2, nlon)
            lon[:] = lon_values
            
            # 创建纬度变量 - 从南到北排列，与原始数据一致
            lat = ds.createVariable('lat', 'f8', ('lat',))
            lat.units = 'degrees_south'
            lat.long_name = 'latitude'
            lat.standard_name = 'latitude'
            lat.axis = 'lat'  # 使用 lon/lat
            lat.valid_min = south
            lat.valid_max = north
            
            # 从南到北排列纬度值 (从低纬度到高纬度) - 与GrADS控制文件一致
            lat_values = np.linspace(north - gsize/2, south + gsize/2, nlat)
            lat[:] = lat_values
        
        # 创建数据变量 - 不包含时间维度
        var = ds.createVariable(var_name, nc_dtype, ('lat', 'lon'), fill_value=fill_value, zlib=True, complevel=4)
        var.description = description or f'Data from {binary_file}'
        var.long_name = description or var_name
        var.standard_name = var_name
        var.units = units or 'unknown'
        var.missing_value = fill_value
        var.coordinates = "lon lat"
        var.grid_mapping = "crs"
        
        # 添加空间参考信息
        crs = ds.createVariable('crs', 'i4')
        crs.grid_mapping_name = "latitude_longitude"
        crs.longitude_of_prime_meridian = 0.0
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        
        # 根据测试结果，需要进行Y轴翻转才能正确显示
        print(f"Data shape: {data.shape}")
        var[:, :] = data#[::-1, :]
        
        # 关闭文件
        ds.close()
        print(f"Successfully converted {binary_file} to {output_file}")
        return True
    except Exception as e:
        print(f"Error converting {binary_file}: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_cama_files(cama_dir, map_name, tag, output_dir, force_overwrite=False):
    """
    转换CaMa-Flood的二进制文件为NetCDF格式
    
    参数:
        cama_dir (str): CaMa-Flood目录
        map_name (str): 地图名称
        tag (str): 标签
        output_dir (str): 输出目录
        force_overwrite (bool): 是否强制覆盖已存在的文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定参数文件路径
    params_file = f"{cama_dir}/{map_name}/params.txt"
    if not os.path.exists(params_file):
        print(f"Error: Parameters file {params_file} not found")
        return False
    
    # 读取参数文件
    try:
        with open(params_file, 'r') as f:
            lines = f.readlines()
            nXX = int(lines[0].split('!')[0].strip())
            nYY = int(lines[1].split('!')[0].strip())
            nFL = int(lines[2].split('!')[0].strip())
            gsize = float(lines[3].split('!')[0].strip())
            west = float(lines[4].split('!')[0].strip())
            east = float(lines[5].split('!')[0].strip())
            south = float(lines[6].split('!')[0].strip())
            north = float(lines[7].split('!')[0].strip())
    except Exception as e:
        print(f"Error reading parameters file {params_file}: {e}")
        return False
    
    print(f"Map parameters: nXX={nXX}, nYY={nYY}, gsize={gsize}")
    print(f"Bounds: west={west}, east={east}, south={south}, north={north}")
    
    # 定义要转换的文件
    files_to_convert = {
        'uparea.bin': (np.float32, 'Upstream area', 'm2'),
        'basin.bin': (np.int32, 'Basin ID', '1'),
        'elevtn.bin': (np.float32, 'Elevation', 'm'),
        'nxtdst.bin': (np.float32, 'Next distance', 'm'),
        'biftag.bin': (np.int32, 'Bifurcation tag', '1')
    }
    
    conversion_success = True
    
    # 转换每个文件
    for filename, (dtype, description, units) in files_to_convert.items():
        input_file = f"{cama_dir}/{map_name}/{filename}"
        
        # 检查文件是否存在
        if not os.path.exists(input_file):
            print(f"Warning: Input file {input_file} not found, skipping")
            continue
            
        output_file = f"{output_dir}/{filename.replace('.bin', '.nc')}"
        var_name = os.path.basename(filename).split('.')[0]
        result = convert_binary_to_netcdf(
            input_file, output_file, (nYY, nXX), dtype, 
            description=description,
            west=west, east=east, south=south, north=north, gsize=gsize,
            var_name=var_name, units=units, force_overwrite=force_overwrite
        )
        if not result:
            conversion_success = False
    
    # 特殊处理nextxy.bin（包含两个记录）
    nextxy_file = f"{cama_dir}/{map_name}/nextxy.bin"
    if os.path.exists(nextxy_file):
        nextxy_nc = f"{output_dir}/nextxy.nc"
        if not convert_nextxy_file(nextxy_file, nextxy_nc, nXX, nYY, west, east, south, north, gsize, force_overwrite=force_overwrite):
            conversion_success = False
    else:
        print(f"Warning: NextXY file {nextxy_file} not found, skipping")
    
    return conversion_success

def convert_hires_files(cama_dir, map_name, tag, output_dir, region=None, file_list=None, force_overwrite=False):
    """
    转换高分辨率二进制文件为NetCDF格式
    
    参数:
        cama_dir (str): CaMa-Flood目录
        map_name (str): 地图名称
        tag (str): 标签
        output_dir (str): 输出目录
        region (str): 区域名称，如果为None则用set_name函数确定
        file_list (list): 需要转换的文件列表，如['catmzz','flddif']，如果为None则转换所有文件
        force_overwrite (bool): 是否强制覆盖已存在的文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查高分辨率目录是否存在
    hires_dir = f"{cama_dir}/{map_name}/{tag}"
    if not os.path.exists(hires_dir):
        print(f"Error: High-resolution directory {hires_dir} not found")
        return False
    
    # 设置分辨率参数
    if tag == "1min":
        hres = 60
        cnum = 60
        mwin = 1
    elif tag == "15sec":
        hres = 4 * 60
        cnum = 240
        mwin = 30
    elif tag == "3sec":
        hres = 20 * 60
        cnum = 1200
        mwin = 10
    elif tag == "1sec":
        hres = 60 * 60
        cnum = 3600
        mwin = 1
    else:
        print(f"Warning: Unknown tag '{tag}', defaulting to 15sec parameters")
        hres = 60
        cnum = 240
        mwin = 30
    
    nx = mwin * cnum
    ny = mwin * cnum
    
    # 确定区域
    if region:
        # 如果直接提供了区域名称
        region_name = region
        # 解析区域名称以确定经纬度范围
        sn = region_name[0]  # 'n' or 's'
        ew = region_name[3]  # 'e' or 'w'
        
        # 解析纬度
        lat_deg = int(region_name[1:3])
        if sn == 's':
            south = -lat_deg
        else:  # 'n'
            south = lat_deg
        
        # 解析经度
        lon_deg = int(region_name[4:7])
        if ew == 'w':
            west = -lon_deg
        else:  # 'e'
            west = lon_deg
    else:
        # 如果没有提供区域名称，尝试查找所有可用区域
        print("No region specified, searching for available regions...")
        import glob
        
        # 尝试不同的目录模式查找区域文件
        region_patterns = [
            f"{hires_dir}/*.bin",  # 直接在目录中的文件
            f"{hires_dir}/*/*.bin"  # 子目录中的文件
        ]
        
        available_regions = set()
        for pattern in region_patterns:
            region_files = glob.glob(pattern)
            for file_path in region_files:
                file_name = os.path.basename(file_path)
                # 尝试从文件名中提取区域名
                parts = file_name.split('.')
                if len(parts) >= 2 and len(parts[0]) == 7 and (parts[0][0] in ['n', 's']) and (parts[0][3] in ['e', 'w']):
                    available_regions.add(parts[0])
        
        # 检查子目录是否有区域名模式
        subdirs = [d for d in os.listdir(hires_dir) if os.path.isdir(os.path.join(hires_dir, d))]
        for subdir in subdirs:
            if len(subdir) == 7 and (subdir[0] in ['n', 's']) and (subdir[3] in ['e', 'w']):
                available_regions.add(subdir)
        
        if available_regions:
            # 按照与s01-allocate_VS.py类似的逻辑，系统地查找最佳区域
            # 优先选择南美洲区域
            south_america_regions = [r for r in available_regions if r.startswith('s') and 'w0' in r]
            if south_america_regions:
                # 优先选择巴西亚马逊地区 (南纬0-10度，西经50-70度)
                amazon_regions = [r for r in south_america_regions if 
                                 (r.startswith('s0') or r.startswith('s1')) and 
                                 ('w05' in r or 'w06' in r or 'w07')]
                if amazon_regions:
                    region_name = sorted(amazon_regions)[0]
                else:
                    region_name = sorted(south_america_regions)[0]
            else:
                # 如果没有南美洲区域，选择任意可用区域
                region_name = sorted(list(available_regions))[0]
            
            print(f"Found available regions: {', '.join(sorted(list(available_regions)))}")
            print(f"Selected region: {region_name}")
            
            # 解析所选区域名称
            sn = region_name[0]  # 'n' or 's'
            ew = region_name[3]  # 'e' or 'w'
            
            # 解析纬度
            lat_deg = int(region_name[1:3])
            if sn == 's':
                south = -lat_deg
            else:  # 'n'
                south = lat_deg
            
            # 解析经度
            lon_deg = int(region_name[4:7])
            if ew == 'w':
                west = -lon_deg
            else:  # 'e'
                west = lon_deg
        else:
            # 如果没有找到可用区域，使用默认的南美洲亚马逊区域
            west = -70
            south = -10
            region_name = set_name(west, south)  # 使用set_name函数，与s01-allocate_VS.py保持一致
            print(f"No available regions found. Using default region: {region_name}")
    
    # 计算区域边界
    north = south + 10  # 每个区域是10度
    east = west + 10
    
    print(f"Processing high-resolution data for region: {region_name}")
    print(f"Region boundaries: west={west}, east={east}, south={south}, north={north}")
    
    # 查找区域文件
    # 首先检查高分辨率目录
    hires_region_dir = f"{hires_dir}/{region_name}"
    region_prefix = f"{region_name}."
    
    # 检查文件存在的可能位置
    data_locations = [
        {"dir": hires_dir, "prefix": region_prefix},  # 如 .../3sec/n10w070.catmzz.bin
        {"dir": hires_dir, "prefix": ""},             # 如 .../3sec/catmzz.bin
        {"dir": hires_region_dir, "prefix": ""}       # 如 .../3sec/n10w070/catmzz.bin
    ]
    
    # 检查该区域是否有任何文件
    any_file_found = False
    test_files = ["catmzz.bin", "flddif.bin", "hand.bin", "elevtn.bin", "uparea.bin", "rivwth.bin"]
    
    for location in data_locations:
        dir_path = location["dir"]
        prefix = location["prefix"]
        if not os.path.exists(dir_path):
            continue
            
        for filename in test_files:
            test_file = f"{dir_path}/{prefix}{filename}"
            if os.path.exists(test_file):
                any_file_found = True
                print(f"Found file: {test_file}")
                break
        if any_file_found:
            current_hires_dir = dir_path
            current_prefix = prefix
            break
    
    if not any_file_found:
        print(f"Warning: No files found for region {region_name}")
        # 尝试在整个目录中查找包含区域名称的文件
        import glob
        pattern = f"{cama_dir}/{map_name}/{tag}/*{region_name}*"
        matching_files = glob.glob(pattern, recursive=True)
        if matching_files:
            print(f"Found files matching pattern: {pattern}")
            for f in matching_files[:5]:  # 只显示前5个匹配文件
                print(f"  - {f}")
                # 如果找到了匹配的文件，确定其所在目录
                if any(f.endswith(test) for test in test_files):
                    current_hires_dir = os.path.dirname(f)
                    current_prefix = f"{region_name}." if region_name in os.path.basename(f) else ""
                    any_file_found = True
                    print(f"Using directory: {current_hires_dir} with prefix: {current_prefix}")
                    break
    
    if not any_file_found:
        print(f"Error: Could not find any valid files for region {region_name}")
        return False
    
    gsize_hires = 10.0 / nx  # 计算网格大小
    
    print(f"High-resolution parameters: nx={nx}, ny={ny}, resolution={gsize_hires}")
    print(f"High-resolution bounds: west={west}, east={east}, south={south}, north={north}")
    
    # 定义要转换的文件
    all_files_to_convert = {
        'catmzz.bin': (np.int8, 'CaMa Z mapping', '1'),
        'flddif.bin': (np.float32, 'Flood depth difference', 'm'),
        'hand.bin': (np.float32, 'Height above nearest drainage', 'm'),
        'elevtn.bin': (np.float32, 'Elevation', 'm'),
        'uparea.bin': (np.float32, 'Upstream area', 'm2'),
        'rivwth.bin': (np.float32, 'River width', 'm'),
        'visual.bin': (np.int8, 'Visualization map', '1'),
        'flwdir.bin': (np.int8, 'Flow direction', '1')
    }
    
    # 如果指定了file_list，则只转换指定的文件
    if file_list:
        files_to_convert = {}
        for file_key in file_list:
            file_key_with_ext = f"{file_key}.bin"
            if file_key_with_ext in all_files_to_convert:
                files_to_convert[file_key_with_ext] = all_files_to_convert[file_key_with_ext]
            else:
                print(f"Warning: Specified file '{file_key}' is not in the known files list")
        if not files_to_convert:
            print("Warning: None of the specified files are in the known files list")
    else:
        # 如果没有指定文件列表，则转换所有文件
        files_to_convert = all_files_to_convert
    
    conversion_success = True
    converted_files = []
    
    # 转换每个文件
    for filename, (dtype, description, units) in files_to_convert.items():
        # 构建可能的文件路径
        input_file = f"{current_hires_dir}/{current_prefix}{filename}"
        
        # 检查文件是否存在
        if not os.path.exists(input_file):
            print(f"Warning: Input file {input_file} not found, skipping")
            continue
            
        output_file = f"{output_dir}/{region_name}.{filename.replace('.bin', '.nc')}"
        var_name = os.path.basename(filename).split('.')[0]
        
        # 对于大型文件，输出内存使用警告
        #if nx * ny > 100000000:  # 1亿个元素，可能约为400MB-1.6GB，取决于数据类型
        #    print(f"Warning: Processing large file {filename} with {nx*ny} elements. Memory usage may be high.")
            
        result = convert_binary_to_netcdf(
            input_file, output_file, (ny, nx), dtype, 
            description=description,
            west=west, east=east, south=south, north=north, 
            gsize=gsize_hires, var_name=var_name, units=units,
            force_overwrite=force_overwrite
        )
        if result:
            converted_files.append(output_file)
        else:
            conversion_success = False
    
    # 特殊处理catmxy.bin（包含两个记录）
    # 只有在没有指定文件列表或文件列表中包含catmxy时才处理
    if not file_list or 'catmxy' in file_list:
        catmxy_file = f"{current_hires_dir}/{current_prefix}catmxy.bin"
        if not os.path.exists(catmxy_file):
            alt_locations = [
                f"{hires_dir}/{region_name}.catmxy.bin",
                f"{hires_dir}/catmxy.bin",
                f"{hires_dir}/{region_name}/catmxy.bin"
            ]
            for alt_file in alt_locations:
                if os.path.exists(alt_file):
                    print(f"Found alternative catmxy file: {alt_file}")
                    catmxy_file = alt_file
                    break
        
        if os.path.exists(catmxy_file):
            catmxy_nc = f"{output_dir}/{region_name}.catmxy.nc"
            if convert_catmxy_file(catmxy_file, catmxy_nc, nx, ny, west, east, south, north, gsize_hires, force_overwrite=force_overwrite):
                converted_files.append(catmxy_nc)
            else:
                conversion_success = False
        else:
            print(f"Warning: CatmXY file not found, skipping")
    
    # 生成区域元数据文件以便后续处理
    if converted_files:
        try:
            metadata_file = f"{output_dir}/{region_name}.metadata.json"
            import json
            metadata = {
                "region": region_name,
                "west": west,
                "east": east,
                "south": south, 
                "north": north,
                "resolution": gsize_hires,
                "nx": nx,
                "ny": ny,
                "files": converted_files
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Generated metadata file: {metadata_file}")
        except Exception as e:
            print(f"Warning: Error generating metadata file: {e}")
    
    return conversion_success

def convert_catmxy_file(catmxy_file, catmxy_nc, nx, ny, west, east, south, north, gsize, force_overwrite=False):
    """
    转换catmxy.bin文件为NetCDF格式
    
    参数:
        force_overwrite (bool): 是否强制覆盖已存在的文件
    """
    # 检查目标文件是否已存在
    if not force_overwrite and os.path.exists(catmxy_nc):
        try:
            # 尝试打开文件检查是否是有效的NetCDF文件
            with nc.Dataset(catmxy_nc, 'r') as ds:
                # 检查变量是否存在
                if 'catmXX' in ds.variables and 'catmYY' in ds.variables:
                    print(f"File {catmxy_nc} already exists and contains required variables. Skipping conversion.")
                    return True
                else:
                    print(f"File {catmxy_nc} exists but does not contain required variables. Will overwrite.")
        except:
            print(f"File {catmxy_nc} exists but is not a valid NetCDF file. Will overwrite.")
    
    try:
        with open(catmxy_file, 'rb') as f:
            # 使用little-endian格式读取数据
            catmXX = np.fromfile(f, dtype=np.int16, count=nx*ny).reshape(ny, nx)
            catmYY = np.fromfile(f, dtype=np.int16, count=nx*ny).reshape(ny, nx)
        
        ds = nc.Dataset(catmxy_nc, 'w', format='NETCDF4')
        ds.title = "CaMa-Flood Catchment XY mapping"
        ds.description = 'Catchment to CaMa XY mapping'
        ds.history = f'Created {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        ds.source = 'CaMa-Flood binary file conversion'
        ds.Conventions = 'CF-1.8'
        ds.institution = "Local Data Processing"
        ds.references = "CaMa-Flood model"
        
        ds.createDimension('lat', ny)
        ds.createDimension('lon', nx)
        
        # 创建坐标变量
        lon = ds.createVariable('lon', 'f8', ('lon',))
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'
        lon.standard_name = 'longitude'
        lon.axis = 'lon'  # 使用 lon/lat
        lon.valid_min = west
        lon.valid_max = east
        lon[:] = np.linspace(west + gsize/2, east - gsize/2, nx)
        
        # 创建纬度变量 - 从南到北排列，与原始数据一致
        lat = ds.createVariable('lat', 'f8', ('lat',))
        lat.units = 'degrees_south'
        lat.long_name = 'latitude'
        lat.standard_name = 'latitude'
        lat.axis = 'lat'  # 使用 lon/lat
        lat.valid_min = south
        lat.valid_max = north
        lat[:] = np.linspace(north - gsize/2, south + gsize/2, ny)
        
        # 添加空间参考信息
        crs = ds.createVariable('crs', 'i4')
        crs.grid_mapping_name = "latitude_longitude"
        crs.longitude_of_prime_meridian = 0.0
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        
        # 创建数据变量 - 不包含时间维度
        # int16的填充值使用-999（范围内）
        fill_value = -999
        var_xx = ds.createVariable('catmXX', 'int32', ('lat', 'lon'), fill_value=fill_value, zlib=True, complevel=4)
        var_yy = ds.createVariable('catmYY', 'int32', ('lat', 'lon'), fill_value=fill_value, zlib=True, complevel=4)
        
        var_xx.long_name = 'Catchment to CaMa X index'
        var_yy.long_name = 'Catchment to CaMa Y index'
        var_xx.standard_name = 'catmXX'
        var_yy.standard_name = 'catmYY'
        var_xx.units = '1'
        var_yy.units = '1'
        var_xx.missing_value = fill_value
        var_yy.missing_value = fill_value
        var_xx.coordinates = "lon lat"
        var_yy.coordinates = "lon lat"
        var_xx.grid_mapping = "crs"
        var_yy.grid_mapping = "crs"
        
        # 需要进行Y轴翻转以正确显示
        var_xx[:, :] = catmXX#[::-1, :]
        var_yy[:, :] = catmYY#[::-1, :]
        
        ds.close()
        print(f"Successfully converted {catmxy_file} to {catmxy_nc}")
        return True
    except Exception as e:
        print(f"Error converting {catmxy_file}: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_nextxy_file(nextxy_file, nextxy_nc, nXX, nYY, west, east, south, north, gsize, force_overwrite=False):
    """
    转换nextxy.bin文件为NetCDF格式
    
    参数:
        force_overwrite (bool): 是否强制覆盖已存在的文件
    """
    # 检查目标文件是否已存在
    if not force_overwrite and os.path.exists(nextxy_nc):
        try:
            # 尝试打开文件检查是否是有效的NetCDF文件
            with nc.Dataset(nextxy_nc, 'r') as ds:
                # 检查变量是否存在
                if 'nextXX' in ds.variables and 'nextYY' in ds.variables:
                    print(f"File {nextxy_nc} already exists and contains required variables. Skipping conversion.")
                    return True
                else:
                    print(f"File {nextxy_nc} exists but does not contain required variables. Will overwrite.")
        except:
            print(f"File {nextxy_nc} exists but is not a valid NetCDF file. Will overwrite.")
    
    try:
        with open(nextxy_file, 'rb') as f:
            nextXX = np.fromfile(f, dtype=np.int32, count=nXX*nYY).reshape(nYY, nXX)
            nextYY = np.fromfile(f, dtype=np.int32, count=nXX*nYY).reshape(nYY, nXX)
        
        ds = nc.Dataset(nextxy_nc, 'w', format='NETCDF4')
        ds.title = "CaMa-Flood Next Downstream Cell"
        ds.description = 'Next downstream cell indices'
        ds.history = f'Created {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        ds.source = 'CaMa-Flood binary file conversion'
        ds.Conventions = 'CF-1.8'
        ds.institution = "Local Data Processing"
        ds.references = "CaMa-Flood model"
        
        ds.createDimension('lat', nYY)
        ds.createDimension('lon', nXX)
        
        # 创建坐标变量，使用linspace确保精确数量的点
        lon = ds.createVariable('lon', 'f8', ('lon',))
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'
        lon.standard_name = 'longitude'
        lon.axis = 'lon'  # 使用 lon/lat
        lon.valid_min = west
        lon.valid_max = east
        lon[:] = np.linspace(west + gsize/2, east - gsize/2, nXX)
        
        # 创建纬度变量 - 从南到北排列，与原始数据一致
        lat = ds.createVariable('lat', 'f8', ('lat',))
        lat.units = 'degrees_south'
        lat.long_name = 'latitude'
        lat.standard_name = 'latitude'
        lat.axis = 'lat'  # 使用 lon/lat
        lat.valid_min = south
        lat.valid_max = north
        lat[:] = np.linspace(north - gsize/2, south + gsize/2, nYY)
        
        # 添加空间参考信息
        crs = ds.createVariable('crs', 'i4')
        crs.grid_mapping_name = "latitude_longitude"
        crs.longitude_of_prime_meridian = 0.0
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        
        # 创建数据变量 - 不包含时间维度
        # int32的填充值可以使用-9999
        fill_value = -9999
        var_xx = ds.createVariable('nextXX', 'int32', ('lat', 'lon'), fill_value=fill_value, zlib=True, complevel=4)
        var_yy = ds.createVariable('nextYY', 'int32', ('lat', 'lon'), fill_value=fill_value, zlib=True, complevel=4)
        
        var_xx.long_name = 'Next downstream cell X index'
        var_yy.long_name = 'Next downstream cell Y index'
        var_xx.standard_name = 'nextXX'
        var_yy.standard_name = 'nextYY'
        var_xx.units = '1'
        var_yy.units = '1'
        var_xx.missing_value = fill_value
        var_yy.missing_value = fill_value
        var_xx.coordinates = "lon lat"
        var_yy.coordinates = "lon lat"
        var_xx.grid_mapping = "crs"
        var_yy.grid_mapping = "crs"
        
        # 需要进行Y轴翻转以正确显示
        var_xx[:, :] = nextXX#[::-1, :]
        var_yy[:, :] = nextYY#[::-1, :]
        
        ds.close()
        print(f"Successfully converted {nextxy_file} to {nextxy_nc}")
        return True
    except Exception as e:
        print(f"Error converting {nextxy_file}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert CaMa-Flood binary files to NetCDF format')
    parser.add_argument('--cama_dir', required=True, help='CaMa-Flood directory')
    parser.add_argument('--map_name', default='glb_15min', help='Map name')
    parser.add_argument('--tag', default='3sec', help='Tag')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--skip_hires', action='store_true', help='Skip high-resolution files conversion')
    parser.add_argument('--skip_cama', action='store_true', help='Skip CaMa files conversion')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--region', help='Region name (e.g., n30e120) for high-resolution data')
    parser.add_argument('--check_s10w070', action='store_true', help='Specifically check for s10w070 region data')
    parser.add_argument('--files', nargs='+', help='Specific high-resolution files to convert (e.g. catmzz flddif hand). If not specified, all files will be converted.')
    parser.add_argument('--force', action='store_true', help='Force overwrite existing NetCDF files')
    
    args = parser.parse_args()
    
    # 验证输入目录
    if not os.path.exists(args.cama_dir):
        print(f"Error: CaMa-Flood directory {args.cama_dir} does not exist")
        sys.exit(1)
    
    success = True
    
    # 如果指定了检查s10w070区域
    if args.check_s10w070:
        region = "s10w070"
        print(f"Specifically checking for {region} region data...")
        if convert_hires_files(args.cama_dir, args.map_name, args.tag, args.output_dir, region=region, file_list=args.files, force_overwrite=args.force):
            print(f"Successfully converted data for region {region}")
        else:
            print(f"Warning: Issues encountered when converting data for region {region}")
        return
    
    # 转换CaMa文件
    if not args.skip_cama:
        print(f"Converting CaMa files with map_name={args.map_name}")
        if not convert_cama_files(args.cama_dir, args.map_name, args.tag, args.output_dir, force_overwrite=args.force):
            print("Warning: Some errors occurred during CaMa file conversion")
            success = False
    
    # 转换高分辨率文件
    if not args.skip_hires:
        if args.files:
            print(f"Converting specific high-resolution files: {args.files} with map_name={args.map_name}, tag={args.tag}, region={args.region}")
        else:
            print(f"Converting all high-resolution files with map_name={args.map_name}, tag={args.tag}, region={args.region}")
        if not convert_hires_files(args.cama_dir, args.map_name, args.tag, args.output_dir, region=args.region, file_list=args.files, force_overwrite=args.force):
            print("Warning: Some errors occurred during high-resolution file conversion")
            success = False
    
    if success:
        print("All conversions completed successfully")
        sys.exit(0)
    else:
        print("Conversion completed with some errors")
        sys.exit(1)

if __name__ == "__main__":
    main() 