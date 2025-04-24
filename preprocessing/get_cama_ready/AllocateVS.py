import numpy as np
import math
from pathlib import Path
import os
import datetime
import netCDF4 as nc
import logging
from typing import List, Tuple, Dict, Union


class AllocateVS:
    """
    将HydroWeb、HydroSat、IICESat VS转换为CaMa-Flood网格的类
    基于Menaka@IIS的allocate_vs.f90 Fortran代码转换而来
    """
    
    def __init__(self, west1=None, south1=None, dataname=None, camadir=None, map_name=None, tag=None, outdir=None):
        """
        初始化AllocateVS类
        
        参数:
            west1 (float): 西边界经度
            south1 (float): 南边界纬度
            dataname (str): 数据集名称
            camadir (str): CaMa-Flood目录路径
            map_name (str): 地图名称，例如: glb_15min
            tag (str): 标签，例如: 1min, 15sec, 3sec
            outdir (str): 输出目录
        """
        # 初始化参数
        self.west1 = west1
        self.south1 = south1
        self.dataname = dataname
        self.camadir = camadir
        self.map = map_name
        self.tag = tag
        self.outdir = outdir
        
        # 基本参数
        self.north1 = None
        self.east1 = None
        
        # CaMa参数
        self.nXX = None
        self.nYY = None
        self.nFL = None
        self.gsize = None
        self.west = None
        self.east = None
        self.south = None
        self.north = None
        
        # 分辨率参数
        self.hres = None
        self.cnum = None
        self.csize = None
        self.mwin = None
        
        # 数据数组
        self.uparea = None
        self.basin = None
        self.elevtn = None
        self.nxtdst = None
        self.nextXX = None
        self.nextYY = None
        self.biftag = None
        
        # 高分辨率数据
        self.nx = None
        self.ny = None
        self.upa1m = None
        self.catmXX = None
        self.catmYY = None
        self.catmZZ = None
        self.dwx1m = None
        self.dwy1m = None
        self.flddif = None
        self.hand = None
        self.ele1m = None
        self.riv1m = None
        self.visual = None
        self.flwdir = None
        
        # 如果参数完整，则初始化
        if all([west1 is not None, south1 is not None, dataname, camadir, map_name, tag, outdir]):
            self.init_params()
            # 检查并转换文件
            if self.check_and_convert_files():
                self.load_cama_data()
                self.load_hires_data()

    def init_params(self):
        """初始化基本参数"""
        # 设置边界
        self.north1 = self.south1 + 10.0
        self.east1 = self.west1 + 10.0
        
        print(f"Debug: Region boundaries:")
        print(f"  west1={self.west1}, east1={self.east1}")
        print(f"  south1={self.south1}, north1={self.north1}")
        
        # 读取CaMa参数
        params_file = f"{self.camadir}/data_for_wse/cama_maps/glb_15min/params.txt"
        print(f"Reading params from: {params_file}")
        try:
            with open(params_file, 'r') as f:
                lines = f.readlines()
                # Strip comments and whitespace for each line
                self.nXX = int(lines[0].split('!!')[0].strip())
                self.nYY = int(lines[1].split('!!')[0].strip())
                self.nFL = int(lines[2].split('!!')[0].strip())
                self.gsize = float(lines[3].split('!!')[0].strip())
                self.west = float(lines[4].split('!!')[0].strip())
                self.east = float(lines[5].split('!!')[0].strip())
                self.south = float(lines[6].split('!!')[0].strip())
                self.north = float(lines[7].split('!!')[0].strip())
                print(f"Loaded CaMa params:")
                print(f"  nXX={self.nXX}, nYY={self.nYY}, nFL={self.nFL}")
                print(f"  gsize={self.gsize}")
                print(f"  west={self.west}, east={self.east}")
                print(f"  south={self.south}, north={self.north}")
        except Exception as e:
            print(f"Error reading params file: {e}")
            return False
            
        # 设置分辨率参数 (完全匹配Fortran代码)
        if self.tag == "1min":
            self.hres = 60
            self.cnum = 60
            self.mwin = 1
        elif self.tag == "15sec":
            self.hres = 4 * 60  # 240
            self.cnum = 240
            self.mwin = 30
        elif self.tag == "3sec":
            self.hres = 20 * 60  # 1200
            self.cnum = 1200
            self.mwin = 10
        elif self.tag == "1sec":
            self.hres = 60 * 60  # 3600
            self.cnum = 3600
            self.mwin = 1
        else:
            self.hres = 60
            self.cnum = 240
            self.mwin = 30
            
        # 计算网格大小 (使用浮点数除法)
        self.csize = 1.0 / float(self.cnum)
            
        print(f"Debug: Resolution parameters:")
        print(f"  tag={self.tag}")
        print(f"  hres={self.hres} (cells per degree)")
        print(f"  cnum={self.cnum}")
        print(f"  csize={self.csize} (degrees)")
        print(f"  mwin={self.mwin}")
            
        # 设置高分辨率数组大小
        self.nx = int(self.mwin * self.cnum)
        self.ny = int(self.mwin * self.cnum)
        
        print(f"Debug: Array dimensions: nx={self.nx}, ny={self.ny}")
        
        # 设置瓦片名 - 确保使用原始经纬度
        if self.tag == "1min":
            self.cname = self.tag
        else:
            # 确保在此使用原始输入的west1和south1，而不是做任何四舍五入或其他调整
            self.cname = self.set_name(self.west1, self.south1)
            
        print(f"Debug: Tile name: {self.cname}")
            
        return True
            
    def load_cama_data(self):
        """
        加载CaMa-Flood数据
        """
        try:
            # 读取参数文件
            params_file = f"{self.camadir}/data_for_wse/cama_maps/glb_15min/params.txt"
            with open(params_file, 'r') as f:
                lines = f.readlines()
                self.nXX = int(lines[0].split('!!')[0].strip())
                self.nYY = int(lines[1].split('!!')[0].strip())
                self.nFL = int(lines[2].split('!!')[0].strip())
                self.gsize = float(lines[3].split('!!')[0].strip())
                self.west = float(lines[4].split('!!')[0].strip())
                self.east = float(lines[5].split('!!')[0].strip())
                self.south = float(lines[6].split('!!')[0].strip())
                self.north = float(lines[7].split('!!')[0].strip())
            
            # 初始化数组
            self.uparea = np.zeros((self.nYY, self.nXX), dtype=np.float32)
            self.basin = np.zeros((self.nYY, self.nXX), dtype=np.int32)
            self.elevtn = np.zeros((self.nYY, self.nXX), dtype=np.float32)
            self.nxtdst = np.zeros((self.nYY, self.nXX), dtype=np.float32)
            self.biftag = np.zeros((self.nYY, self.nXX), dtype=np.int32)
            self.nextXX = np.zeros((self.nYY, self.nXX), dtype=np.int32)
            self.nextYY = np.zeros((self.nYY, self.nXX), dtype=np.int32)
            
            # 从NetCDF文件加载数据
            base_path = f"{self.camadir}/data_for_wse/cama_maps/glb_15min"
            
            # 加载uparea
            with nc.Dataset(f"{base_path}/uparea.nc", 'r') as ds:
                self.uparea = ds.variables['uparea'][:].filled(-9999)
            
            # 加载basin
            with nc.Dataset(f"{base_path}/basin.nc", 'r') as ds:
                self.basin = ds.variables['basin'][:].filled(-9999)
            
            # 加载elevtn
            with nc.Dataset(f"{base_path}/elevtn.nc", 'r') as ds:
                self.elevtn = ds.variables['elevtn'][:].filled(-9999)
            print(f"Debug: elevtn shape: {self.elevtn.shape}")  
            # 加载nxtdst
            with nc.Dataset(f"{base_path}/nxtdst.nc", 'r') as ds:
                self.nxtdst = ds.variables['nxtdst'][:].filled(-9999)
            
            # 加载biftag
            with nc.Dataset(f"{base_path}/biftag.nc", 'r') as ds:
                self.biftag = ds.variables['biftag'][:].filled(-9999)
            
            # 加载nextxy
            with nc.Dataset(f"{base_path}/nextxy.nc", 'r') as ds:
                self.nextXX = ds.variables['nextXX'][:].filled(-9999)
                self.nextYY = ds.variables['nextYY'][:].filled(-9999)
            print(f"Debug: nextxy shape: {self.nextXX.shape}")
            print(f"Debug: nextYY shape: {self.nextYY.shape}")
            
            print("Successfully loaded CaMa-Flood data")
            return True
        except Exception as e:
            print(f"Error loading CaMa-Flood data: {e}")
            return False

    def load_hires_data(self):
        """
        加载高分辨率数据
        """
        try:
            # 确保使用正确的区域名称 (由init_params中设置的self.cname)
            print(f"Loading high-resolution data for region: {self.cname}")
            print(f"Original coordinates: west1={self.west1}, south1={self.south1}")
            
            # 获取正确的瓦片路径和文件前缀
            base_path = f"{self.camadir}/data_for_wse/cama_map/{self.map}/{self.tag}"
            file_prefix = self.cname
            print(f"Loading high-resolution data from: {base_path}")
            print(f"Using file prefix: {file_prefix}")
            
            # 计算备用区域名称（直接使用整数值）
            alt_prefix = None
            if self.west1 != int(self.west1) or self.south1 != int(self.south1):
                int_west = int(self.west1)
                int_south = int(self.south1)
                alt_prefix = self.set_name(int_west, int_south)
                print(f"Calculated alternate region name: {alt_prefix}")
            
            # 初始化高分辨率数组
            self.catmXX = np.zeros((self.ny, self.nx), dtype=np.int32)
            self.catmYY = np.zeros((self.ny, self.nx), dtype=np.int32)
            self.catmZZ = np.zeros((self.ny, self.nx), dtype=np.int32)
            self.flddif = np.zeros((self.ny, self.nx), dtype=np.float32)
            self.hand = np.zeros((self.ny, self.nx), dtype=np.float32)
            self.ele1m = np.zeros((self.ny, self.nx), dtype=np.float32)
            self.riv1m = np.zeros((self.ny, self.nx), dtype=np.float32)
            self.visual = np.zeros((self.ny, self.nx), dtype=np.int32)
            self.flwdir = np.zeros((self.ny, self.nx), dtype=np.int32)
            self.upa1m = np.zeros((self.ny, self.nx), dtype=np.float32)
            
            # 尝试加载所有文件
            success = True
            
            def load_file_with_fallback(file_type):
                """尝试加载文件，如果原始区域名称不存在则尝试备用区域名称"""
                primary_file = f"{base_path}/{file_prefix}.{file_type}.nc"
                if os.path.exists(primary_file):
                    return primary_file
                
                # 如果原始文件不存在且有备用名称，尝试备用名称
                if alt_prefix and alt_prefix != file_prefix:
                    alt_file = f"{base_path}/{alt_prefix}.{file_type}.nc"
                    if os.path.exists(alt_file):
                        print(f"Using alternate file: {alt_file}")
                        return alt_file
                
                # 如果还是没找到，尝试对文件中字符做大小写不敏感查找
                possible_files = []
                for f in os.listdir(base_path):
                    if f.lower().endswith(f"_{file_type.lower()}.nc"):
                        possible_files.append(f)
                
                if possible_files:
                    print(f"Found possible alternative files for {file_type}: {possible_files}")
                    return f"{base_path}/{possible_files[0]}"
                
                # 如果都没找到，返回原始文件名
                return primary_file
            
            # 加载catmxy
            try:
                catmxy_file = load_file_with_fallback("catmxy")
                with nc.Dataset(catmxy_file, 'r') as ds:
                    self.catmXX = ds.variables['catmXX'][:].filled(-9999)
                    self.catmYY = ds.variables['catmYY'][:].filled(-9999)
            except Exception as e:
                print(f"Error loading catmxy: {e}")
                success = False
            
            # 加载catmzz
            try:
                catmzz_file = load_file_with_fallback("catmzz")
                with nc.Dataset(catmzz_file, 'r') as ds:
                    self.catmZZ = ds.variables['catmzz'][:].filled(-9999)
            except Exception as e:
                print(f"Error loading catmzz: {e}")
                success = False
            
            # 加载flddif
            try:
                flddif_file = load_file_with_fallback("flddif")
                with nc.Dataset(flddif_file, 'r') as ds:
                    self.flddif = ds.variables['flddif'][:].filled(-9999)
            except Exception as e:
                print(f"Error loading flddif: {e}")
                success = False
            
            # 加载hand
            try:
                hand_file = load_file_with_fallback("hand")
                with nc.Dataset(hand_file, 'r') as ds:
                    self.hand = ds.variables['hand'][:].filled(-9999)
            except Exception as e:
                print(f"Error loading hand: {e}")
                success = False
            
            # 加载elevtn (高分辨率)
            try:
                elevtn_file = load_file_with_fallback("elevtn")
                with nc.Dataset(elevtn_file, 'r') as ds:
                    self.ele1m = ds.variables['elevtn'][:].filled(-9999)
            except Exception as e:
                print(f"Error loading elevtn: {e}")
                success = False
            
            # 加载uparea (高分辨率)
            try:
                uparea_file = load_file_with_fallback("uparea")
                with nc.Dataset(uparea_file, 'r') as ds:
                    self.upa1m = ds.variables['uparea'][:].filled(-9999)
            except Exception as e:
                print(f"Error loading uparea: {e}")
                success = False
            
            # 加载rivwth
            try:
                rivwth_file = load_file_with_fallback("rivwth")
                with nc.Dataset(rivwth_file, 'r') as ds:
                    self.riv1m = ds.variables['rivwth'][:].filled(-9999)
            except Exception as e:
                print(f"Error loading rivwth: {e}")
                success = False
            
            # 加载visual
            try:
                visual_file = load_file_with_fallback("visual")
                with nc.Dataset(visual_file, 'r') as ds:
                    self.visual = ds.variables['visual'][:].filled(-9999)
            except Exception as e:
                print(f"Error loading visual: {e}")
                success = False
            
            # 加载flwdir
            try:
                flwdir_file = load_file_with_fallback("flwdir")
                with nc.Dataset(flwdir_file, 'r') as ds:
                    self.flwdir = ds.variables['flwdir'][:].filled(-9999)
            except Exception as e:
                print(f"Error loading flwdir: {e}")
                success = False
            
            if success:
                print("Successfully loaded high-resolution data")
            else:
                print("Warning: Some high-resolution data files could not be loaded")
            
            return success
        except Exception as e:
            print(f"Error loading high-resolution data: {e}")
            # 输出更多调试信息来帮助诊断问题
            print(f"Debug info: west1={self.west1}, south1={self.south1}, cname={self.cname}")
            import traceback
            traceback.print_exc()
            return False

    def _load_netcdf_file(self, filename, array, var_name):
        """加载NetCDF文件到数组"""
        try:
            with nc.Dataset(filename, 'r') as ds:
                np.copyto(array, ds.variables[var_name][:])
            self._add_file_loading_log(filename, "success", array.dtype)
            self.check_file_structure(filename, array, array.dtype)
            return True
        except Exception as e:
            error_msg = f"Error loading {filename}: {e}"
            print(error_msg)
            self._add_file_loading_log(filename, f"failed: {e}", array.dtype)
            return False
            
    def check_file_structure(self, filename, array, dtype):
        """检查文件结构并保存统计信息"""
        try:
            # 只对一部分文件进行详细检查
            if 'visual' in filename or 'catmxy' in filename or 'rivwth' in filename:
                # 创建诊断目录
                diag_dir = f"{self.outdir}/file_diagnostics"
                os.makedirs(diag_dir, exist_ok=True)
                
                # 提取基本文件名
                base_name = os.path.basename(filename)
                diag_file = f"{diag_dir}/{base_name}_diagnostics.txt"
                
                with open(diag_file, 'w') as f:
                    # 写入基本信息
                    f.write(f"File: {filename}\n")
                    f.write(f"Data type: {dtype}\n")
                    f.write(f"Array shape: {array.shape}\n")
                    
                    # 计算统计信息
                    if array.size > 0:
                        f.write(f"Data range: [{np.min(array)}, {np.max(array)}]\n")
                        f.write(f"Mean: {np.mean(array)}\n")
                        f.write(f"Non-zero count: {np.count_nonzero(array)}\n")
                        f.write(f"Zero count: {array.size - np.count_nonzero(array)}\n")
                        
                        # 保存值分布情况
                        unique_values, counts = np.unique(array, return_counts=True)
                        f.write(f"Unique values count: {len(unique_values)}\n")
                        
                        # 只显示前20个唯一值的计数
                        if len(unique_values) <= 20:
                            for val, count in zip(unique_values, counts):
                                f.write(f"  Value {val}: {count} ({count/array.size*100:.2f}%)\n")
                        else:
                            f.write("Top 20 most common values:\n")
                            # 获取最常见的20个值
                            top_indices = np.argsort(counts)[-20:]
                            for i in reversed(top_indices):
                                f.write(f"  Value {unique_values[i]}: {counts[i]} ({counts[i]/array.size*100:.2f}%)\n")
                
                # 如果是visual或rivwth，保存一个简单的诊断图像
                if 'visual' in filename or 'rivwth' in filename:
                    try:
                        import matplotlib.pyplot as plt
                        
                        plt.figure(figsize=(10, 8))
                        
                        # 对于visual，使用离散颜色
                        if 'visual' in filename:
                            plt.imshow(array, cmap='viridis', interpolation='none')
                            plt.colorbar(label='Visual value')
                            plt.title(f'Visual map for {self.cname}')
                        
                        # 对于rivwth，使用连续颜色
                        elif 'rivwth' in filename:
                            # 创建掩码，显示河流宽度>0的区域
                            masked_data = np.ma.masked_where(array <= 0, array)
                            plt.imshow(masked_data, cmap='Blues', interpolation='none')
                            plt.colorbar(label='River width')
                            plt.title(f'River width for {self.cname}')
                        
                        plt.savefig(f"{diag_dir}/{base_name}_preview.png")
                        plt.close()
                    except Exception as e:
                        print(f"Error creating diagnostic image: {e}")
                        
        except Exception as e:
            print(f"Error in check_file_structure: {e}")
            
    def _add_file_loading_log(self, filename, status, dtype):
        """记录文件加载状态"""
        try:
            log_dir = f"{self.outdir}/file_load_logs"
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = f"{log_dir}/file_loading_log_{self.cname}.txt"
            with open(log_file, 'a') as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} | {filename} | {dtype} | {status}\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")
            
    def set_name(self, lon, lat):
        """设置瓦片名称"""
        # 确保使用整数部分对经纬度进行处理
        int_lon = int(lon)
        int_lat = int(lat)
        
        print(f"set_name called with lon={lon}, lat={lat}")
        print(f"Using integer parts: int_lon={int_lon}, int_lat={int_lat}")
        
        if int_lon < 0:
            ew = 'w'
            clon = f"{int(-int_lon):03d}"
        else:
            ew = 'e'
            clon = f"{int_lon:03d}"
            
        if int_lat < 0:
            sn = 's'
            clat = f"{int(-int_lat):02d}"
        else:
            sn = 'n'
            clat = f"{int_lat:02d}"
        
        cname = f"{sn}{clat}{ew}{clon}"
        print(f"Generated region name: {cname}")
        return cname
        
    def westsouth(self, lon, lat):
        """获取最近的西边界和南边界"""
        if lon > 0.0:
            west = int(lon - (lon % self.mwin))
        else:
            lon1 = abs(lon)
            west = -(int(lon1 - (lon1 % self.mwin)) + 10.0)
            
        if lat > 0.0:
            south = int(lat - (lat % self.mwin))
        else:
            lat1 = abs(lat)
            south = -(int(lat1 - (lat1 % self.mwin)) + 10.0)
            
        return west, south

    def roundx(self, ix):
        """循环网格索引"""
        if ix >= 1:
            return ix - int((ix - 1) / self.nx) * self.nx
        else:
            return self.nx - abs(ix % self.nx)
            
    def ixy2iixy(self, ix, iy, nx, ny):
        """处理边界条件的网格索引变换"""
        if iy < 1:
            iiy = 2 - iy
            iix = ix + int(nx / 2.0)
            iix = self.roundx(iix)
        elif iy > ny:
            iiy = 2 * ny - iy
            iix = ix + int(nx / 2.0)
            iix = self.roundx(iix)
        else:
            iiy = iy
            iix = self.roundx(ix)
            
        return iix, iiy
        
    def hubeny_real(self, lat1, lon1, lat2, lon2):
        """计算两点之间的距离（米）"""
        pi = math.pi
        a = 6378137.0
        b = 6356752.314140
        e2 = 0.00669438002301188
        a_1_e2 = 6335439.32708317
        
        latrad1 = lat1 * pi / 180.0
        latrad2 = lat2 * pi / 180.0
        lonrad1 = lon1 * pi / 180.0
        lonrad2 = lon2 * pi / 180.0
        
        latave = (latrad1 + latrad2) / 2.0
        dlat = latrad2 - latrad1
        dlon = lonrad2 - lonrad1
        
        dlondeg = lon2 - lon1
        if abs(dlondeg) > 180.0:
            dlondeg = 180.0 - (abs(dlondeg) % 180.0)
            dlon = dlondeg * pi / 180.0
            
        W = math.sqrt(1.0 - e2 * math.sin(latave)**2.0)
        M = a_1_e2 / (W**3.0)
        N = a / W
        
        return math.sqrt((dlat * M)**2.0 + (dlon * N * math.cos(latave))**2.0)
        
    def D8(self, dx, dy):
        """确定D8方向编号"""
        # D8 示意图:
        # |-----------|
        # | 8 | 1 | 2 |
        # |-----------|
        # | 7 | 0 | 3 |
        # |-----------|
        # | 6 | 5 | 4 |
        # |-----------|
        
        if dx == 0:
            if dy > 0:
                return 5
            if dy < 0:
                return 1
        elif dy == 0:
            if dx > 0:
                return 3
            if dx < 0:
                return 7
        elif dx > 0 and dy > 0:
            tval = abs(float(dy) / float(dx))
            if 0 < tval <= 0.4142135:
                return 4 
            if 0.4142135 < tval <= 2.4142135:
                return 5
            if tval > 2.4142135:
                return 6
        elif dx > 0 and dy < 0:
            tval = abs(float(dy) / float(dx))
            if 0 < tval <= 0.4142135:
                return 3 
            if 0.4142135 < tval <= 2.4142135:
                return 2
            if tval > 2.4142135:
                return 1
        elif dx < 0 and dy > 0:
            tval = abs(float(dy) / float(dx))
            if 0 < tval <= 0.4142135:
                return 7 
            if 0.4142135 < tval <= 2.4142135:
                return 6
            if tval > 2.4142135:
                return 5
        elif dx < 0 and dy < 0:
            tval = abs(float(dy) / float(dx))
            if 0 < tval <= 0.4142135:
                return 7
            if 0.4142135 < tval <= 2.4142135:
                return 8
            if tval > 2.4142135:
                return 1
                
        return 0  # 默认情况
        
    def next_D8(self, dval):
        """根据D8方向获取下一个像素的偏移量
        D8 示意图:
        |-----------|
        | 8 | 1 | 2 |
        |-----------|
        | 7 | 0 | 3 |
        |-----------|
        | 6 | 5 | 4 |
        |-----------|
        """
        if dval == 1:
            return 0, -1
        if dval == 2:
            return 1, -1
        if dval == 3:
            return 1, 0
        if dval == 4:
            return 1, 1
        if dval == 5:
            return 0, 1
        if dval == 6:
            return -1, 1
        if dval == 7:
            return -1, 0
        if dval == 8:
            return -1, -1
            
        return 0, 0  # 默认情况

    def upstream(self, i, j):
        """寻找流域上游像素，选择流域面积最接近的像素"""
        # 在CaMa网格中找到上游像素
        x = -9999
        y = -9999
        d = 10  # 搜索框大小
        dA = 1.0e20  # 流域面积差异初始值
        
        for tx in range(i-d, i+d+1):
            for ty in range(j-d, j+d+1):
                ix, iy = self.ixy2iixy(tx, ty, self.nXX, self.nYY)
                # 如果该点流向当前点
                if self.nextXX[iy-1, ix-1] == i and self.nextYY[iy-1, ix-1] == j:
                    # 如果流域面积差异更小
                    if self.uparea[iy-1, ix-1] - self.uparea[j-1, i-1] < dA:
                        dA = self.uparea[iy-1, ix-1] - self.uparea[j-1, i-1]
                        x = ix
                        y = iy
        
        return x, y
    
    def next_pixel(self, tx, ty):
        """获取下一个像素的位置"""
        dval = self.flwdir[ty-1, tx-1]
        dx, dy = self.next_D8(dval)
        ix = tx + dx
        iy = ty + dy
        return ix, iy
    
    def hires_upstream(self, i, j):
        """在高分辨率网格中寻找上游像素"""
        x = -9999
        y = -9999
        d = 3  # 搜索框大小
        dA = 1.0e20  # 流域面积差异初始值
        
        for tx in range(i-d, i+d+1):
            for ty in range(j-d, j+d+1):
                if tx < 1 or tx > self.nx or ty < 1 or ty > self.ny:
                    continue
                if tx == i and ty == j:
                    continue
                if self.visual[ty-1, tx-1] == 10 or self.visual[ty-1, tx-1] == 20:
                    ix, iy = self.next_pixel(tx, ty)
                    if ix == i and iy == j:
                        if abs(self.upa1m[ty-1, tx-1] - self.upa1m[j-1, i-1]) < dA:
                            dA = abs(self.upa1m[j-1, i-1] - self.upa1m[ty-1, tx-1])
                            x = tx
                            y = ty
    
        return x, y
    
    def up_until_mouth(self, ix, iy):
        """寻找上游单元流域出口"""
        x = -9999
        y = -9999
        
        iix = ix
        iiy = iy
        
        if iix == -9999 or iiy == -9999:
            return x, y
        
        while self.visual[iiy-1, iix-1] == 10:
            if iix == -9999 or iiy == -9999:
                x = -9999
                y = -9999
                break
                
            x0, y0 = self.hires_upstream(iix, iiy)
            iix = x0
            iiy = y0
            
            if iix == -9999 or iiy == -9999:
                break
                
            if self.visual[iiy-1, iix-1] == 20:  # 找到上游单元流域出口
                x = x0
                y = y0
                break
        
        return x, y
    
    def unit_catchment_mouth(self, ix, iy):
        """寻找单元流域出口"""
        x0 = ix
        y0 = iy
        
        iix = ix
        iiy = iy
        
        while self.visual[iiy-1, iix-1] == 10:
            if iix < 1 or iiy < 1 or iix > self.nx or iiy > self.ny:
                break
                
            dval = self.flwdir[iiy-1, iix-1]
            dx, dy = self.next_D8(dval)
            iix = iix + dx
            iiy = iiy + dy
            
            if iix < 1 or iiy < 1 or iix > self.nx or iiy > self.ny:
                break
                
            if self.flwdir[iiy-1, iix-1] == -9:  # 河口
                break
                
            if self.visual[iiy-1, iix-1] == 2:  # 陆地
                break
                
            if self.visual[iiy-1, iix-1] == 20:  # 单元流域出口
                x0 = iix
                y0 = iiy
                break
                
            if self.visual[iiy-1, iix-1] == 25:  # 单元流域出口
                x0 = iix
                y0 = iiy
                break
        
        return x0, y0
     
    def down_dist(self, ix, iy):
        """计算下游距离"""
        iix = ix 
        iiy = iy
        count = 0
        down_dist = 0.0
        west0 = self.west1+ self.csize/2.0
        north0 = self.north1-self.csize/2.0
        print(west0, north0)
        #lon1 = self.west1 + (ix+0.5)*(1/float(self.hres))
        #lat1 = self.north1 - (iy+0.5)*(1/float(self.hres))
        lon1 = west0+ix*self.csize
        lat1 = north0-iy*self.csize
        print(f"  lon1, lat1: {lon1}, {lat1}")

        while self.visual[iiy-1, iix-1] != 20:
            if count > 1000:
                down_dist = -9999.0
                break
                
            if iix < 1 or iiy < 1 or iix > self.nx or iiy > self.ny:
                break
                
            if self.flwdir[iiy-1, iix-1] == -9:  # 河口
                break
                
            if self.visual[iiy-1, iix-1] == 2:  # 陆地
                break
                
            if self.visual[iiy-1, iix-1] == 20:  # 单元流域出口
                break
                
            dval = self.flwdir[iiy-1, iix-1]
            print(f"  dval: {dval}")
            dx, dy = self.next_D8(dval)
            iix = iix + dx
            iiy = iiy + dy
            
            if iix < 1 or iiy < 1 or iix > self.nx or iiy > self.ny:
                break
                
            lon2=lon1+dx*self.csize 
            lat2=lat1+dy*self.csize

            down_dist = down_dist + self.hubeny_real(lat1, lon1, lat2, lon2)
            
            lon1 = lon2
            lat1 = lat2
            count = count + 1
        
        if count > 1000:
            down_dist = -9999.0
        print(f"  lon2, lat2: {lon2}, {lat2}")
        print(f"  down_dist: {down_dist}")
        return down_dist
    
    def until_mouth_flag(self, ix, iy, x0, y0):
        """检查两点是否在同一条河流上"""
        flag = 0
        iix = ix
        iiy = iy
        
        while self.visual[iiy-1, iix-1] == 10:
            if iix < 1 or iiy < 1 or iix > self.nx or iiy > self.ny:
                break
                
            dval = self.flwdir[iiy-1, iix-1]
            dx, dy = self.next_D8(dval)
            iix = iix + dx
            iiy = iiy + dy
            
            if self.flwdir[iiy-1, iix-1] == -9:  # 河口
                break
                
            if self.visual[iiy-1, iix-1] == 2:  # 非河道
                break
                
            if self.visual[iiy-1, iix-1] == 20:  # 单元流域出口
                dval = self.flwdir[iiy-1, iix-1]
                dx, dy = self.next_D8(dval)
                iix = iix + dx
                iiy = iiy + dy
                
            if self.visual[iix, iiy] == 25:  # 河口出口
                break
                
            if iix < 1 or iiy < 1 or iix > self.nx or iiy > self.ny:
                break
                
            if iix == x0 and iiy == y0:
                flag = 1
                break
        
        return flag
    
    def find_nearest_river(self, ix, iy):
        """寻找最近的河流中心线"""
        kx = -9999
        ky = -9999
        lag = 1.0e20
        
        # 搜索范围
        nn = 20
        
        # 在搜索范围内寻找最近的河流中心线
        for dy in range(-nn, nn+1):
            for dx in range(-nn, nn+1):
                jx = ix + dx
                jy = iy + dy
                
                if jx <= 0 or jx > self.nx or jy <= 0 or jy > self.ny:
                    continue
                    
                if ix == jx and iy == jy:
                    continue
                    
                if self.visual[jy-1, jx-1] == 10:  # 河流中心线
                    dx = ix - jx
                    dy = iy - jy
                    lag_now = math.sqrt(dx**2 + dy**2)
                    if lag_now < lag:
                        lag = lag_now
                        kx = jx
                        ky = jy
        
        return kx, ky, lag
    
    def find_nearest_main_river(self, ix, iy):
        """寻找最近的主河道"""
        kx = -9999
        ky = -9999
        nn = 60  # 搜索范围
        lag = 1.0e20
        lag_now = 1.0e20
        
        # 首先找到最大流域面积的位置
        iix = ix
        iiy = iy
        uparea_max = 0.0
        
        for dy in range(-nn, nn+1):
            for dx in range(-nn, nn+1):
                jx = ix + dx
                jy = iy + dy
                
                if jx <= 0 or jx > self.nx or jy <= 0 or jy > self.ny:
                    continue
                    
                if ix == jx and iy == jy:
                    continue
                    
                if self.catmXX[jy-1, jx-1] <= 0 or self.catmYY[jy-1, jx-1] <= 0:
                    continue
                    
                if self.visual[jy-1, jx-1] == 10 or self.visual[jy-1, jx-1] == 20:
                    if self.upa1m[jy-1, jx-1] > uparea_max:
                        uparea_max = self.upa1m[jy-1, jx-1]
                        iix = jx
                        iiy = jy
        
        # 寻找上游单元流域出口
        if iix != -9999 and iiy != -9999:
            x0, y0 = self.up_until_mouth(iix, iiy)
        else:
            x0 = -9999
            y0 = -9999
            
        if x0 == -9999 or y0 == -9999:
            uparea_max = 0.01 * uparea_max
        else:
            uparea_max = self.upa1m[x0, y0]
            
        # 寻找符合条件的最近主河道
        for dy in range(-nn, nn+1):
            for dx in range(-nn, nn+1):
                jx = ix + dx
                jy = iy + dy
                
                if jx <= 0 or jx > self.nx or jy <= 0 or jy > self.ny:
                    continue
                    
                lag_now = math.sqrt(dx**2 + dy**2)
                
                # 检查是否在同一条河流上
                flag = self.until_mouth_flag(ix, iy, jx, jy)
                if flag == 1:
                    continue
                    
                if ix == jx and iy == jy:
                    continue
                    
                if lag_now == -9999.0:
                    continue
                    
                if self.upa1m[jx, jy] < self.upa1m[ix, iy]:
                    continue
                    
                if self.upa1m[jx, jy] < uparea_max:
                    continue
                    
                if lag_now < lag:
                    if self.visual[jx, jy] == 10:
                        kx = jx
                        ky = jy
                        lag = lag_now
        
        return kx, ky, lag
    
    def perpendicular_grid(self, ix, iy):
        """获取垂直于河流的网格点列表"""
        xlist = [-9] * 100
        ylist = [-9] * 100
        
        # 寻找单元流域出口
        x0, y0 = self.unit_catchment_mouth(ix, iy)
        
        dx = x0 - ix 
        dy = y0 - iy
        
        if dx == 0 and dy == 0:
            dval = self.flwdir[iy-1, ix-1]
        else:
            dval = self.D8(dx, dy)
            
        k = int(self.riv1m[iy-1, ix-1] / 90.0) + 100
        k = max(k, 100)
        j = 1
        
        # 根据流向确定垂直方向
        if dval == 1 or dval == 5:
            for i in range(-k, k+1):
                iix = ix + i
                iiy = iy
                if iix < 0 or iix >= self.nx or iiy < 0 or iiy >= self.ny:
                    continue
                if self.visual[iiy-1, iix-1] == 10 or self.visual[iiy-1, iix-1] == 20:
                    xlist[j] = iix
                    ylist[j] = iiy
                    j += 1
        elif dval == 3 or dval == 7:
            for i in range(-k, k+1):
                iix = ix
                iiy = iy + i
                if iix < 0 or iix >= self.nx or iiy < 0 or iiy >= self.ny:
                    continue
                if self.visual[iiy-1, iix-1] == 10 or self.visual[iiy-1, iix-1] == 20:
                    xlist[j] = iix
                    ylist[j] = iiy
                    j += 1
        elif dval == 4 or dval == 8:
            for i in range(-k, k+1):
                iix = ix - i
                iiy = iy + i
                if iix < 0 or iix >= self.nx or iiy < 0 or iiy >= self.ny:
                    continue
                if self.visual[iiy-1, iix-1] == 10 or self.visual[iiy-1, iix-1] == 20:
                    xlist[j] = iix
                    ylist[j] = iiy
                    j += 1
        elif dval == 2 or dval == 6:
            for i in range(-k, k+1):
                iix = ix + i
                iiy = iy + i
                if iix < 0 or iix >= self.nx or iiy < 0 or iiy >= self.ny:
                    continue
                if self.visual[iiy-1, iix-1] == 10 or self.visual[iiy-1, iix-1] == 20:
                    xlist[j] = iix
                    ylist[j] = iiy
                    j += 1
        
        return xlist, ylist, j-1
    
    def find_nearest_main_river_ppend(self, ix, iy):
        """寻找垂直方向上的最近主河道"""
        # 获取垂直于河流的网格点列表
        xlist, ylist, k = self.perpendicular_grid(ix, iy)
        
        lag = 1.0e20
        kx = -9999
        ky = -9999
        iix = ix
        iiy = iy
        uparea_max = 0.0
        for i in range(1, k+1):
            jx = xlist[i]
            jy = ylist[i]
            
            if jx < 0 or jx >= self.nx or jy < 0 or jy >= self.ny:
                continue
                
            # 检查是否在同一条河上
            flag = self.until_mouth_flag(ix, iy, jx, jy)
            if flag == 1:
                continue
                
            if self.visual[jx, jy] < 10:  # 非水体区域
                continue
                
            if ix == jx and iy == jy:  # 相同位置
                continue
                
            if self.catmXX[jx, jy] <= 0 or self.catmYY[jx, jy] <= 0:
                continue
                
            if self.upa1m[jx, jy] < self.upa1m[ix, iy]:  # 上游区域面积更小
                continue
                
            if self.upa1m[jx, jy] > uparea_max:
                dx = ix - jx
                dy = iy - jy
                lag_now = math.sqrt(dx**2 + dy**2)
                
                if lag_now > lag:
                    continue
                    
                uparea_max = self.upa1m[jx, jy]
                kx = jx
                ky = jy
                lag = lag_now
        
        return kx, ky, lag
    
    def process_station(self, id, station, river, bsn, country, lon0, lat0, ele0, egm08, egm96, sat, stime, etime, status):
        """处理单个站点的数据"""
        # 检查站点是否在当前区域内
        if lon0 < self.west1 or lon0 > self.east1 or lat0 < self.south1 or lat0 > self.north1:
            return None
            
        # 计算中间值 - 和Fortran代码保持一致
        lon_term = lon0 - self.west1 - self.csize/2.0
        lat_term = self.north1 - self.csize/2.0 - lat0
        
        # 高精度计算站点索引 
        ix = int(lon_term * self.hres) + 1 #1-based
        iy = int(lat_term * self.hres) + 1 #1-based
        
        print(f"Debug: Calculated HiRes Index: ix={ix}, iy={iy}")
        #print the lon lat of ix, iy
        lon_hres = self.west1 + (ix)*(1/float(self.hres))
        lat_hres = self.north1 - (iy)*(1/float(self.hres))
        print(f"Debug: HiRes Index (ix, iy) -> Lon Lat: ({lon_hres:.5f}, {lat_hres:.5f})")

        print(f"[DEBUG] Station {station} ({lon0:.5f}, {lat0:.5f}) -> Initial HiRes Index (ix, iy): ({ix}, {iy}) within tile {self.cname}")
        #print the lon lat of ix, iy
        lon_hres = self.west1 + (ix)*(1/float(self.hres))
        lat_hres = self.north1 - (iy)*(1/float(self.hres))
        print(f"Debug: HiRes Index (ix, iy) -> Lon Lat: ({lon_hres:.2f}, {lat_hres:.2f})")

        # 检查索引是否有效
        if ix < 1 or ix > self.nx or iy < 1 or iy > self.ny:
            print(f"Warning: Invalid grid indices for station {station}: ix={ix}, iy={iy} (Tile dimensions: nx={self.nx}, ny={self.ny})")
            return None
        
        # 标记代码说明
        # 10 = 在河流中心线上
        # 11 = 在河道内
        # 12 = 位置在单元流域出口
        # 20 = 找到最近的河流
        # 21 = 找到最近的主河流
        # 30 = 找到最近垂直的主河流
        # 31 = 分叉位置
        # 40 = 对海洋网格进行校正
        
        flag = -9
        kx1 = -9999  # 初始化为-9999，与Fortran代码保持一致
        ky1 = -9999
        kx2 = -9999
        ky2 = -9999
        kx1 = ix  # 然后设置为当前位置
        ky1 = iy
        
        lat1 = lat0
        lon1 = lon0
        
        print(f"Debug: Processing station {station}")
        print(f"  Position: ix={ix}, iy={iy}")
        print(f"  visual[{iy-1},{ix-1}]={self.visual[iy-1,ix-1]}")
        print(f"  riv1m[{iy-1},{ix-1}]={self.riv1m[iy-1,ix-1]}")
        
        # 检查点的类型并处理 (注意：数组索引需要-1转换为0-based)
        if self.visual[iy-1, ix-1] == 10:  # 河流中心线
            print(f"  Station {station} is on river centerline")
            kx1 = ix
            ky1 = iy
            kx2 = -9999
            ky2 = -9999
            flag = 10
        elif self.riv1m[iy-1, ix-1] == -1:  # 河道
            print(f"  Station {station} is in river channel")
            # 找最近的河流中心线
            kx1, ky1, lag1 = self.find_nearest_river(ix, iy)
            kx2 = -9999
            ky2 = -9999
            flag = 11
        elif self.visual[iy-1, ix-1] == 20:  # 单元流域出口
            print(f"  Station {station} is at unit-catchment outlet")
            kx1 = ix
            ky1 = iy
            kx2 = -9999
            ky2 = -9999
            flag = 12
        # 陆地/网格单元、边界到河道的校正
        elif self.visual[iy-1, ix-1] in [2, 3, 5, 7]:
            print(f"  Station {station} needs correction from land to channel")
            # 找最近的河流
            kx2, ky2, lag2 = self.find_nearest_river(ix, iy)
            if kx2 == -9999 or ky2 == -9999:
                kx2 = ix
                ky2 = iy
            # 找垂直于河流的最近主河道
            kx1, ky1, lag1 = self.find_nearest_main_river_ppend(kx2, ky2)
            if kx1 == -9999 or ky1 == -9999:
                # 尝试找最近的主河道
                kx3, ky3, lag3 = self.find_nearest_main_river(kx2, ky2)
                if kx3 == -9999 or ky3 == -9999:
                    kx1 = kx2
                    ky1 = ky2
                    kx2 = -9999
                    ky2 = -9999
                    flag = 20
                else:
                    kx1 = kx3
                    ky1 = ky3
                    flag = 21
            else:
                # 检查是否找到了一个新的位置
                if kx1 != kx2 or ky1 != ky2:
                    if lag2 < lag1:
                        flag = 30
                    else:
                        kx2 = -9999
                        ky2 = -9999
                        flag = 20
                else:
                    kx2 = -9999
                    ky2 = -9999
                    flag = 20
        # 海洋网格校正
        elif self.visual[iy-1, ix-1] in [0, 1, 25]:
            print(f"  Station {station} needs ocean grid correction")
            kx1, ky1, lag1 = self.find_nearest_river(ix, iy)
            if kx1 != ix or ky1 != iy:
                kx2 = -9999
                ky2 = -9999
                flag = 40
            else:
                flag = 90
        else:
            flag = 90
            
        # Print results after finding river point
        print(f"[DEBUG] After river finding: flag={flag}, kx1={kx1}, ky1={ky1}, kx2={kx2}, ky2={ky2}")
        
        # 使用找到的最佳位置
        kx = kx1
        ky = ky1
 
        # 检查索引有效性
        if kx < 1 or ky < 1 or kx > self.nx or ky > self.ny:
            print(f"Warning: Invalid grid indices after processing for station {station}: kx={kx}, ky={ky}")
            return None
        
        # 获取CaMa网格索引 (注意：数组索引需要-1转换为0-based)
        # Remove yrev correction - use direct indices instead
        # 检查数组是否越界
        if kx-1 < 0 or kx-1 >= self.nx or ky-1 < 0 or ky-1 >= self.ny:
            print(f"Warning: Array indices out of bounds for catmXX/catmYY: kx-1={kx-1}, ky-1={ky-1}, array shape=({self.nx}, {self.ny})")
            return None

        # 打印catmXX和catmYY的值
        print(f"Debug: catmXX[{ky-1},{kx-1}]={self.catmXX[ky-1, kx-1]}")
        print(f"Debug: catmYY[{ky-1},{kx-1}]={self.catmYY[ky-1, kx-1]}")
        # 获取CaMa网格索引
        # Fortran数组是列优先(column-major)，而Python是行优先(row-major)
        # 在catmXX和catmYY中，第一个索引是y(行)，第二个是x(列)
        iXX = self.catmXX[ky-1, kx-1]   
        iYY = self.catmYY[ky-1, kx-1]    
        print(f"  iXX, iYY: {iXX}, {iYY}")
        print(f"[DEBUG] Mapped to CaMa Index (iXX, iYY) using catm files at (kx, ky)=({kx}, {ky}): ({iXX}, {iYY})")

        # Check if this is a bifurcation location
        if self.biftag[iYY-1, iXX-1] == 1:
            # Check if we have a secondary point and it's different from the primary point
            if kx2 != -9999 or ky2 != -9999:
                if kx1 != kx2 or ky1 != ky2:
                    # This is a bifurcation location
                    print("Bifurcation location detected")
                    flag = 31
        kx=kx1
        ky=ky1
        print(f"  kx, ky: {kx}, {ky}")
        if kx < 1 or ky < 1 or kx > self.nx or ky > self.ny:
            print(f"Warning: Invalid grid indices after processing for station {station}: kx={kx}, ky={ky}")
            return None
     
        # 计算下游距离 (注意：数组索引需要-1转换为0-based)
        if self.visual[ky-1, kx-1] == 10:
            diffdist = self.down_dist(kx, ky)
        else:
            diffdist = 0.0
        print(f"  diffdist: {diffdist}")
        if diffdist == -9999.0:
            print(f"Warning: Invalid downstream distance for station {station}")
            return None
            
        # 计算与原始位置的距离
        dist1 = -9999.0
        dist2 = -9999.0
        
        lat2 = self.north1 - (ky)*(1/float(self.hres))
        lon2 = self.west1 + (kx)*(1/float(self.hres))
        dist1 = self.hubeny_real(lat1, lon1, lat2, lon2)
        
        if kx2 != -9999 or ky2 != -9999:
            lat2 = self.north1 - (ky2)*(1/float(self.hres))
            lon2 = self.west1 + (kx2)*(1/float(self.hres))
            dist2 = self.hubeny_real(lat1, lon1, lat2, lon2)
            
        # 获取最终的CaMa网格索引 (注意：数组索引需要-1转换为0-based)
        iXX = self.catmXX[ky-1, kx-1]
        iYY = self.catmYY[ky-1, kx-1]
        print(f"Debug: self.elevtn shape: {self.elevtn.shape}")
        print(f"Debug: self.ele1m shape: {self.ele1m.shape}")
        print(f"Debug: iXX, iYY: {iXX}, {iYY}")
        uXX, uYY = self.upstream(iXX, iYY)
        print(f"Debug: uXX, uYY: {uXX}, {uYY}")
        
        # 比较上下游高程差，选取合适的网格
        try:
            uXX, uYY = self.upstream(iXX, iYY)
            
            # 检查上游索引有效性
            if uXX < 1 or uXX > self.nXX or uYY < 1 or uYY > self.nYY:
                print(f"Warning: Invalid upstream indices for station {station}: uXX={uXX}, uYY={uYY}")
                # 使用当前索引作为上游索引
                uXX = iXX
                uYY = iYY
            
            # 检查索引是否在有效范围内
            if not (0 <= iYY-1 < self.nYY and 0 <= iXX-1 < self.nXX and 
                    0 <= uYY-1 < self.nYY and 0 <= uXX-1 < self.nXX):
                print(f"Warning: CaMa grid indices out of bounds: iXX={iXX}, iYY={iYY}, uXX={uXX}, uYY={uYY}")
                print(f"Valid ranges: 1 to {self.nXX} for X, 1 to {self.nYY} for Y")
                # Skip the elevation comparison
                raise ValueError("CaMa grid indices out of bounds")
                
            # 注意：数组索引需要-1转换为0-based
            delv0 = abs(self.elevtn[iYY-1, iXX-1] - self.ele1m[ky-1, kx-1])  # 与下游的高程差
            delv1 = abs(self.elevtn[uYY-1, uXX-1] - self.ele1m[ky-1, kx-1])  # 与上游的高程差
            
            # 如果上游高程差更小，则分配到上游网格
            iXX_orig, iYY_orig = iXX, iYY # Store original before potential upstream adjustment
            if delv1 < delv0:
                iXX = uXX
                iYY = uYY
                flag = flag + 100
            
            # Print result after upstream adjustment check
            if iXX != iXX_orig or iYY != iYY_orig:
                print(f"[DEBUG] Adjusted to upstream CaMa Index (iXX, iYY): ({iXX}, {iYY}) due to elevation diff (delv0={delv0:.2f}, delv1={delv1:.2f}). Flag changed to {flag}.")
            else:
                print(f"[DEBUG] Kept original CaMa Index (iXX, iYY): ({iXX}, {iYY}) after elevation check (delv0={delv0:.2f}, delv1={delv1:.2f})")
        except Exception as e:
            print(f"Warning: Error during upstream elevation comparison: {e}")
            # Continue without upstream adjustment
        
        # 计算CaMa网格中心坐标
        lon_cama = self.west + (iXX - 0.5) * self.gsize
        lat_cama = self.north - (iYY - 0.5) * self.gsize
        
        print(f"[DEBUG] Final CaMa Grid Center (lon_cama, lat_cama): ({lon_cama:.2f}, {lat_cama:.2f}) for final (iXX, iYY)=({iXX}, {iYY})")

        # 构建结果字典，确保保存egm96_final
        result = {
            'id': id,
            'station': station,
            'dataname': self.dataname,
            'lon': lon0,
            'lat': lat0,
            'sat': sat,
            'flag': flag,
            'ele': self.ele1m[ky-1, kx-1],
            'diffdist': diffdist * 1e-3,
            'kx1': kx1,
            'ky1': ky1,
            'kx2': kx2,
            'ky2': ky2,
            'dist1': dist1,
            'dist2': dist2,
            'rivwth': self.riv1m[ky-1, kx-1],
            'iXX': iXX,
            'iYY': iYY,
            'egm08': egm08,
            'egm96': egm96,  # 使用计算值或输入值
            'lon_cama': lon_cama,
            'lat_cama': lat_cama
        }
        
        return result
        
    def process_stations(self):
        """处理所有站点数据"""
        # 读取站点列表
        rlist = f"./SampleStation_list.txt"
        results = []
        
        try:
            with open(rlist, 'r') as f:
                # 跳过标题行
                next(f)
                
                # 逐行处理站点数据
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 14:
                        continue
                        
                    id = parts[0]
                    station = parts[1]
                    river = parts[2]
                    bsn = parts[3]
                    country = parts[4]
                    lon0 = float(parts[5])
                    lat0 = float(parts[6])
                    ele0 = float(parts[7])
                    egm08 = float(parts[8])
                    egm96 = float(parts[9])
                    sat = parts[10]
                    stime = parts[11]
                    etime = parts[12]
                    status = parts[13]
                    
                    # 处理单个站点
                    result = self.process_station(id, station, river, bsn, country, lon0, lat0, ele0, 
                                                egm08, egm96, sat, stime, etime, status)
                    
                    if result:
                        results.append(result)
                        print(
                            f"{result['id']:13s} {result['station']:60s} {result['dataname']:10s} "
                            f"{result['lon']:10.2f} {result['lat']:10.2f} {result['sat']:15s} "
                            f"{result['flag']:4d} {result['ele']:10.2f} {result['diffdist']:13.2f} "
                            f"{result['kx1']:8d} {result['ky1']:8d} {result['kx2']:8d} {result['ky2']:8d} "
                            f"{result['dist1']:12.2f} {result['dist2']:12.2f} {result['rivwth']:12.2f} "
                            f"{result['iXX']:8d} {result['iYY']:8d} {result['egm08']:10.2f} {result['egm96']:10.2f}\n"
                        )
        except Exception as e:
            print(f"Error processing station list: {e}")
            import traceback
            traceback.print_exc()
            
        return results
    
    def save_results(self, results):
        """保存处理结果到文件"""
        if not results:
            print("No results to save")
            return
            
        output_file = f"{self.outdir}/altimetry_{self.dataname}_{self.map}.txt"
        
        try:
            with open(output_file, 'w') as f:
                # 写入标题行 - 格式与s01-allocate_VS.py保持一致
                f.write(
                    f"{'ID':13}{'station':64}{'dataname':12}{'lon':12}{'lat':10}"
                    f"{'satellite':17}{'flag':6}{'elevation':12}{'dist_to_mouth':15}"
                    f"{'kx1':10}{'ky1':8}{'kx2':8}{'ky2':8}{'dist1':14}{'dist2':12}"
                    f"{'rivwth':12}{'ix':10}{'iy':8}{'Lon_CaMa':12}{'Lat_CaMa':12}"
                    f"{'EGM08':12}{'EGM96':10}\n"
                )
                
                # 写入数据行
                for result in results:
                    f.write(
                        f"{result['id']:13}{result['station']:64}{result['dataname']:12}"
                        f"{result['lon']:12.2f}{result['lat']:10.2f}{result['sat']:17}"
                        f"{result['flag']:6d}{result['ele']:12.2f}{result['diffdist']:15.2f}"
                        f"{result['kx1']:10d}{result['ky1']:8d}{result['kx2']:8d}{result['ky2']:8d}"
                        f"{result['dist1']:14.2f}{result['dist2']:12.2f}{result['rivwth']:12.2f}"
                        f"{result['iXX']:10d}{result['iYY']:8d}"
                        f"{result['lon_cama']:12.2f}{result['lat_cama']:12.2f}"
                        f"{result['egm08']:12.2f}{result['egm96']:10.2f}\n"
                    )
                    
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def save_to_netcdf(self, data_arrays):
        """保存数据为NetCDF格式"""
        try:
            import netCDF4 as nc
            import datetime
            
            # 创建NetCDF文件
            nc_file = f"{self.outdir}/cama_data_{self.cname}.nc"
            dataset = nc.Dataset(nc_file, 'w', format='NETCDF4')
            
            # 添加全局属性
            dataset.description = f'CaMa-Flood data for region {self.cname}'
            dataset.history = f'Created {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            dataset.source = 'AllocateVS processing'
            dataset.west = self.west1
            dataset.east = self.east1
            dataset.south = self.south1
            dataset.north = self.north1
            
            # 添加坐标轴
            dataset.createDimension('x', self.nx)
            dataset.createDimension('y', self.ny)
            
            # 创建经纬度变量
            lon = dataset.createVariable('lon', 'f4', ('x',))
            lat = dataset.createVariable('lat', 'f4', ('y',))
            
            # 设置经纬度值
            lon.units = 'degrees_east'
            lat.units = 'degrees_north'
            lon[:] = np.linspace(self.west1, self.east1, self.nx)
            lat[:] = np.linspace(self.north1, self.south1, self.ny)
            
            # 记录哪些文件成功加载
            loaded_files = dataset.createVariable('loaded_files', 'i1')
            loaded_files.description = 'Flag indicating which files were successfully loaded'
            
            # 创建变量
            variables_to_save = {
                'catmXX': (self.catmXX, 'i2', 'CaMa X mapping'),
                'catmYY': (self.catmYY, 'i2', 'CaMa Y mapping'),
                'catmZZ': (self.catmZZ, 'i1', 'CaMa Z mapping'),
                'flddif': (self.flddif, 'f4', 'Flood depth difference'),
                'hand': (self.hand, 'f4', 'Height above nearest drainage'),
                'ele1m': (self.ele1m, 'f4', 'Elevation'),
                'visual': (self.visual, 'i1', 'Visualization map'),
                'riv1m': (self.riv1m, 'f4', 'River width'),
                'flwdir': (self.flwdir, 'i1', 'Flow direction')
            }
            
            # 保存变量
            for name, (data, dtype, desc) in variables_to_save.items():
                if data is not None:
                    try:
                        var = dataset.createVariable(name, dtype, ('y', 'x'))
                        var.description = desc
                        var[:] = data
                        print(f"Saved {name} to NetCDF")
                    except Exception as e:
                        print(f"Error saving {name} to NetCDF: {e}")
            
            # 保存一个简化的可视化数据，用于检查
            if self.visual is not None:
                try:
                    simple_visual = np.zeros((self.ny, self.nx), dtype=np.int8)
                    # 河流中心线为1，单元流域出口为2，其他为0
                    simple_visual[self.visual == 10] = 1
                    simple_visual[self.visual == 20] = 2
                    
                    var = dataset.createVariable('simple_visual', 'i1', ('y', 'x'))
                    var.description = 'Simplified visualization (1=river, 2=outlet, 0=other)'
                    var[:] = simple_visual
                except Exception as e:
                    print(f"Error saving simplified visual: {e}")
            
            # 保存统计信息
            try:
                stats = dataset.createVariable('stats', 'i4')
                stats.description = 'Basic statistics'
                
                # 统计河流像素数量
                if self.visual is not None:
                    river_pixels = np.sum(self.visual == 10)
                    outlet_pixels = np.sum(self.visual == 20)
                    stats.river_pixels = river_pixels
                    stats.outlet_pixels = outlet_pixels
                    print(f"Region {self.cname} has {river_pixels} river pixels and {outlet_pixels} outlet pixels")
            except Exception as e:
                print(f"Error saving statistics: {e}")
            
            # 关闭NetCDF文件
            dataset.close()
            print(f"Data saved to NetCDF file: {nc_file}")
            return nc_file
        except ImportError:
            print("Error: netCDF4 module is required for saving to NetCDF format")
            print("Please install with: pip install netCDF4")
            return None
        except Exception as e:
            print(f"Error saving to NetCDF: {e}")
            return None
        
    def find_downstream_points(self, iXX, iYY):
        """找到下游点的位置"""
        next_iXX = self.nextXX[iYY-1, iXX-1]
        next_iYY = self.nextYY[iYY-1, iXX-1]
        
        if next_iXX <= 0 or next_iYY <= 0:
            return None, None
        
        return next_iXX, next_iYY

    def check_and_convert_files(self):
        """
        检查NetCDF文件是否存在，如果不存在则从二进制文件转换
        现在会根据SampleStation_list.txt中的站点位置来检查
        """
        try:
            # 检查CaMa-Flood基础文件
            base_path = f"{self.camadir}/data_for_wse/cama_maps/glb_15min"
            base_files = ['uparea.nc', 'basin.nc', 'elevtn.nc', 'nxtdst.nc', 'biftag.nc', 'nextxy.nc']
            
            # 首先检查基础文件是否存在
            base_missing = False
            for file in base_files:
                file_path = f"{base_path}/{file}"
                if not os.path.exists(file_path):
                    print(f"Missing base file: {file_path}")
                    base_missing = True
            
            if base_missing:
                print("Converting CaMa-Flood base files...")
                # 导入转换函数
                from bin_to_netcdf import convert_cama_files
                # 转换CaMa-Flood基础文件
                convert_cama_files(self.camadir, self.map, self.tag, base_path)
            
            # 读取SampleStation_list.txt找到所有站点
            station_regions = set()
            station_regions.add(self.cname)  # 确保包含当前瓦片
            
            try:
                print(f"Reading SampleStation_list.txt to find relevant regions...")
                with open("./SampleStation_list.txt", 'r') as f:
                    # 跳过标题行
                    next(f)
                    
                    # 读取站点数据
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 14:
                            continue
                            
                        lon0 = float(parts[5])
                        lat0 = float(parts[6])
                        
                        # 计算站点所在的10度区域
                        if lon0 < 0:
                            west = int(lon0 / 10) * 10
                        else:
                            west = int(lon0 / 10) * 10
                        
                        if lat0 < 0:
                            south = int(lat0 / 10) * 10
                        else:
                            south = int(lat0 / 10) * 10
                        
                        # 获取区域名称
                        if self.tag == "1min":
                            region = self.tag
                        else:
                            region = self.set_name(west, south)
                        
                        # 检查该点是否在当前瓦片内
                        if (self.west1 <= lon0 < self.west1 + 10 and 
                            self.south1 <= lat0 < self.south1 + 10):
                            print(f"Station at ({lon0}, {lat0}) is in current tile {self.cname}")
                        
                        station_regions.add(region)
            except Exception as e:
                print(f"Error reading station list: {e}")
                # 如果无法读取站点列表，则使用当前区域
                station_regions.add(self.cname)
            
            # 确保当前区域已添加到集合中
            if self.cname not in station_regions:
                print(f"Warning: Current region {self.cname} not found in station regions. Adding it.")
                station_regions.add(self.cname)
            
            print(f"Found stations in regions: {station_regions}")
            
            # 检查每个站点所在区域的高分辨率文件
            hires_missing = {}
            for region in station_regions:
                hires_path = f"{self.camadir}/data_for_wse/cama_map/{self.map}/{self.tag}"
                hires_files = [
                    f"{region}.catmxy.nc", 
                    f"{region}.catmzz.nc", 
                    f"{region}.flddif.nc", 
                    f"{region}.hand.nc", 
                    f"{region}.elevtn.nc", 
                    f"{region}.uparea.nc", 
                    f"{region}.rivwth.nc", 
                    f"{region}.visual.nc", 
                    f"{region}.flwdir.nc"
                ]
                
                missing = []
                for file in hires_files:
                    file_path = f"{hires_path}/{file}"
                    if not os.path.exists(file_path):
                        missing.append(file)
                
                if missing:
                    hires_missing[region] = missing
                    if region == self.cname:
                        print(f"Warning: Current region {region} missing files: {missing}")
                    else:
                        print(f"Region {region} missing files: {missing}")
            
            # 检查特别针对当前瓦片
            if self.cname in hires_missing:
                print(f"Current region {self.cname} missing these files:")
                for file in hires_missing[self.cname]:
                    print(f"  - {file}")
            
            # 转换缺失的高分辨率文件
            if hires_missing:
                print("Converting high-resolution files for missing regions...")
                # 导入转换函数
                from bin_to_netcdf import convert_hires_files
                
                for region, missing in hires_missing.items():
                    if region == self.cname:
                        print(f"Converting files for current region {region}...")
                    else:
                        print(f"Converting files for region {region}...")
                    hires_path = f"{self.camadir}/data_for_wse/cama_map/{self.map}/{self.tag}"
                    # 转换高分辨率文件
                    convert_hires_files(self.camadir, self.map, self.tag, hires_path, region)
            
            print("File check and conversion completed.")
            
            # 检查EGM96文件
            if not self.egm96_path.exists():
                print(f"警告: EGM96文件不存在: {self.egm96_path}")
                print("EGM96高度计算将不可用")
            else:
                print(f"EGM96文件存在: {self.egm96_path}")
            
            return True
        except Exception as e:
            print(f"Error checking/converting files: {e}")
            return False

def main():
    """主函数"""
    import sys
    import os
    
    # 检查命令行参数
    #if len(sys.argv) < 8:
    #    print("Usage: python set_map.py west1 south1 dataname camadir map tag outdir")
   #     sys.exit(1)
        
    # 解析命令行参数
    west1 = -180.0
    south1 = -30.0
    dataname = "hydroweb"
    camadir = "./data_for_wse"
    map_name = "glb_15min"
    tag = "3sec"
    outdir = "./"
    
    # 创建输出目录
    os.makedirs(outdir, exist_ok=True)
    
    # 实例化并运行处理过程
    allocateVS = AllocateVS(west1, south1, dataname, camadir, map_name, tag, outdir)
    results = allocateVS.process_stations()
    allocateVS.save_results(results)

if __name__ == "__main__":
    main()
