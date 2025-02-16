import os
import glob
import xarray as xr
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# 定义根目录
root_dir = "/share/home/dq013/zhwei/OpenBench/20250120/data/simulation/single_point_test"  # 替换为实际路径
output_dir = "./output"  # 输出 CSV 文件的目录

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

print(f"开始处理根目录: {root_dir}")
print(f"输出目录设置为: {output_dir}")

# 首先获取所有可能的 scenarios
all_scenarios = set()
for site_id in os.listdir(root_dir):
    site_dir = os.path.join(root_dir, site_id)
    if os.path.isdir(site_dir):
        all_scenarios.update([s for s in os.listdir(site_dir) if os.path.isdir(os.path.join(site_dir, s))])

print(f"发现 {len(all_scenarios)} 个不同的场景")

def process_site(args):
    site_id, site_dir, scenario, scenario_dir = args
    try:
        print(f"正在处理站点: {site_id}")  # 添加站点处理状态
        # 检查 history 文件夹
        history_dir = os.path.join(scenario_dir, "history")
        if not os.path.isdir(history_dir):
            print(f"跳过站点 {site_id}: 未找到 history 文件夹")
            return None

        # 获取所有历史文件
        history_files = sorted(glob.glob(os.path.join(history_dir, "*.nc")))
        if not history_files:
            print(f"跳过站点 {site_id}: 未找到 nc 文件")
            return None

        print(f"站点 {site_id}: 发现 {len(history_files)} 个历史文件")

        # 只读取第一个文件获取经纬度信息
        with xr.open_dataset(history_files[0]) as ds:
            lon = float(ds["lon"].values)
            lat = float(ds["lat"].values)

        # 使用 xr.open_mfdataset 一次性读取所有文件
        merged_ds = xr.open_mfdataset(history_files, combine='by_coords')
        
        # 提取年份范围
        years = [int(f.split("_")[-1].split(".")[0]) for f in history_files]
        syear, eyear = min(years), max(years)

        # 创建合并后的文件名和路径
        merged_filename = f"sim_{site_id}_{syear}_{eyear}.nc"
        merged_filepath = os.path.join(scenario_dir, merged_filename)
        
        # 保存合并后的文件
        merged_ds.to_netcdf(merged_filepath)
        merged_ds.close()  # 显式关闭数据集

        return {
            "scenario": scenario,
            "ID": site_id,
            "SYEAR": syear,
            "EYEAR": eyear,
            "LON": lon,
            "LAT": lat,
            "DIR": merged_filepath,
        }
    except Exception as e:
        print(f"处理站点 {site_id} 时出错: {str(e)}")
        return None

# 对每个 scenario 进行处理
for i, scenario in enumerate(sorted(all_scenarios), 1):
    print(f"\n处理场景 {i}/{len(all_scenarios)}: {scenario}")
    
    # 准备并行处理的参数
    process_args = []
    valid_sites = 0
    for site_id in os.listdir(root_dir):
        site_dir = os.path.join(root_dir, site_id)
        if not os.path.isdir(site_dir):
            continue
        
        scenario_dir = os.path.join(site_dir, scenario)
        if not os.path.isdir(scenario_dir):
            continue
            
        process_args.append((site_id, site_dir, scenario, scenario_dir))
        valid_sites += 1
    
    print(f"找到 {valid_sites} 个有效站点待处理")

    # 使用进程池并行处理
    scenario_data = []
    max_workers = multiprocessing.cpu_count()
    print(f"使用 {max_workers} 个进程进行并行处理")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_site, process_args)
        
        # 收集结果
        for result in results:
            if result is not None:
                scenario_data.append(result)

    # 处理完所有站点后，保存这个场景的 CSV 文件
    if scenario_data:
        df = pd.DataFrame(scenario_data)
        output_csv = os.path.join(output_dir, f"{scenario}.csv")
        df.to_csv(output_csv, index=False)
        print(f"已生成 {scenario}.csv，包含 {len(scenario_data)} 个站点")

print("\n所有场景处理完成！")