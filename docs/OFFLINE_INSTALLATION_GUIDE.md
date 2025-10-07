# OpenBench 离线安装指南

本指南将帮助您创建OpenBench的离线安装包,以便在无网络环境下部署。

## 一、检测到的依赖包清单

### 1.1 核心依赖包 (必需)

根据代码分析,OpenBench依赖以下Python包:

```txt
# 数据处理
numpy>=1.21.0
pandas>=1.3.0
xarray>=0.19.0
netCDF4>=1.5.7
scipy>=1.7.0

# 可视化
matplotlib>=3.4.0
cartopy>=0.20.0

# 并行处理
joblib>=1.1.0
dask>=2022.1.0
flox>=0.5.0

# 报告生成
jinja2>=3.0.0
xhtml2pdf>=0.2.5

# 其他工具
tqdm
PyYAML
shapely
cmaps (可选,用于colormap)
```

### 1.2 GUI依赖包 (可选)

如果需要使用GUI界面:

```txt
streamlit>=1.20.0
streamlit-option-menu>=0.3.0
```

### 1.3 系统依赖

某些包需要系统级依赖:

- **GEOS** (用于cartopy和shapely)
- **PROJ** (用于cartopy)
- **HDF5** (用于netCDF4)
- **netCDF C库** (用于netCDF4)

## 二、制作离线安装包

### 方法1: 使用 pip download (推荐)

#### 步骤1: 在有网络的机器上下载所有依赖

```bash
# 创建下载目录
mkdir openbench-offline-packages
cd openbench-offline-packages

# 下载核心依赖包
pip download -r /path/to/OpenBench/requirements.txt -d ./core-packages

# 如果需要GUI,下载GUI依赖
pip download streamlit streamlit-option-menu -d ./gui-packages

# 下载所有依赖(包括间接依赖)
pip download -r /path/to/OpenBench/requirements.txt \
              streamlit streamlit-option-menu \
              -d ./all-packages
```

#### 步骤2: 打包

```bash
# 返回上级目录
cd ..

# 将OpenBench源码和下载的包一起打包
tar -czf openbench-offline-install.tar.gz \
    OpenBench/ \
    openbench-offline-packages/
```

#### 步骤3: 在离线机器上安装

```bash
# 解压
tar -xzf openbench-offline-install.tar.gz
cd openbench-offline-packages

# 安装核心包
pip install --no-index --find-links=./all-packages -r ../OpenBench/requirements.txt

# 如果需要GUI
pip install --no-index --find-links=./all-packages streamlit streamlit-option-menu

# 安装OpenBench
cd ../OpenBench
pip install -e .
```

### 方法2: 使用 conda-pack (适合Conda环境)

#### 步骤1: 在有网络的机器上创建并打包环境

```bash
# 创建新的conda环境
conda create -n openbench python=3.10

# 激活环境
conda activate openbench

# 安装依赖
conda install -c conda-forge numpy pandas xarray netCDF4 scipy \
                              matplotlib cartopy joblib dask \
                              jinja2 PyYAML shapely tqdm flox

pip install xhtml2pdf cmaps

# 如果需要GUI
conda install -c conda-forge streamlit
pip install streamlit-option-menu

# 安装conda-pack
conda install -c conda-forge conda-pack

# 打包环境
conda pack -n openbench -o openbench-env.tar.gz
```

#### 步骤2: 将打包的环境和OpenBench源码一起传输

```bash
# 创建完整的离线安装包
tar -czf openbench-complete-offline.tar.gz \
    openbench-env.tar.gz \
    OpenBench/
```

#### 步骤3: 在离线机器上部署

```bash
# 解压
tar -xzf openbench-complete-offline.tar.gz

# 创建conda环境目录
mkdir -p ~/miniconda3/envs/openbench

# 解压conda环境
tar -xzf openbench-env.tar.gz -C ~/miniconda3/envs/openbench

# 激活环境
source ~/miniconda3/envs/openbench/bin/activate

# 清理路径(可选但推荐)
conda-unpack

# 安装OpenBench
cd OpenBench
pip install -e .
```

### 方法3: 使用 Docker (最可靠)

#### 步骤1: 创建Dockerfile

创建文件 `OpenBench/Dockerfile`:

```dockerfile
FROM continuumio/miniconda3:latest

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libgeos-dev \
    libproj-dev \
    libhdf5-dev \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# 复制requirements
COPY requirements.txt .

# 安装Python依赖
RUN conda install -c conda-forge --file requirements.txt -y && \
    conda clean -a -y

# 安装GUI依赖(可选)
RUN pip install streamlit streamlit-option-menu

# 复制OpenBench代码
COPY . .

# 安装OpenBench
RUN pip install -e .

# 暴露端口(如果使用GUI)
EXPOSE 8501

# 设置入口点
ENTRYPOINT ["python", "openbench/openbench.py"]
```

#### 步骤2: 构建Docker镜像

```bash
cd OpenBench
docker build -t openbench:latest .
```

#### 步骤3: 保存Docker镜像

```bash
# 保存镜像为tar文件
docker save openbench:latest -o openbench-docker.tar

# 压缩
gzip openbench-docker.tar
```

#### 步骤4: 在离线机器上加载

```bash
# 解压
gunzip openbench-docker.tar.gz

# 加载镜像
docker load -i openbench-docker.tar

# 运行
docker run -v /path/to/data:/data openbench:latest /data/config.json
```

## 三、针对不同操作系统的建议

### 3.1 Linux (推荐方法: conda-pack 或 Docker)

**优点**:
- conda-pack: 环境完全一致,包含所有系统依赖
- Docker: 最可靠,完全隔离

**步骤**:
```bash
# 使用conda-pack
1. 在联网机器上: conda pack -n openbench -o openbench-linux.tar.gz
2. 传输到离线机器
3. 解压并激活环境
```

### 3.2 macOS (推荐方法: conda-pack)

**注意事项**:
- macOS上编译的包不能直接用于Linux
- 需要在相同的macOS版本上打包和部署

**步骤**:
```bash
# 确保架构匹配(Intel vs Apple Silicon)
uname -m  # 查看架构

# 使用conda-pack
conda pack -n openbench -o openbench-macos-$(uname -m).tar.gz
```

### 3.3 Windows (推荐方法: pip download + wheels)

**步骤**:
```powershell
# 下载Windows wheels
pip download -r requirements.txt -d .\openbench-windows-packages --platform win_amd64

# 在离线机器安装
pip install --no-index --find-links=.\openbench-windows-packages -r requirements.txt
```

## 四、完整的离线安装包清单

### 4.1 标准配置(不含GUI)

```
openbench-offline-standard/
├── OpenBench/                    # 源代码
│   ├── openbench/
│   ├── requirements.txt
│   ├── setup.py
│   └── ...
├── packages/                     # Python包
│   ├── numpy-*.whl
│   ├── pandas-*.whl
│   ├── xarray-*.whl
│   └── ...
├── install.sh                    # Linux/macOS安装脚本
├── install.bat                   # Windows安装脚本
└── README_OFFLINE.txt           # 离线安装说明
```

### 4.2 完整配置(含GUI)

```
openbench-offline-full/
├── OpenBench/                    # 源代码
├── packages/                     # 所有Python包
├── conda-env/                    # Conda环境(可选)
│   └── openbench-env.tar.gz
├── docker/                       # Docker镜像(可选)
│   └── openbench-docker.tar.gz
├── install.sh
├── install.bat
└── README_OFFLINE.txt
```

## 五、创建自动化安装脚本

### 5.1 Linux/macOS安装脚本

创建 `install.sh`:

```bash
#!/bin/bash
set -e

echo "OpenBench Offline Installation"
echo "=============================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $python_version"

# 安装包
echo "Installing packages from local directory..."
pip3 install --no-index --find-links=./packages -r OpenBench/requirements.txt

# 安装OpenBench
echo "Installing OpenBench..."
cd OpenBench
pip3 install -e .
cd ..

echo ""
echo "Installation completed!"
echo "Run: python3 -m openbench.openbench --help"
```

### 5.2 Windows安装脚本

创建 `install.bat`:

```batch
@echo off
echo OpenBench Offline Installation
echo ==============================

REM Check Python
python --version
if errorlevel 1 (
    echo Python not found! Please install Python 3.10+
    exit /b 1
)

REM Install packages
echo Installing packages from local directory...
pip install --no-index --find-links=packages -r OpenBench\requirements.txt

REM Install OpenBench
echo Installing OpenBench...
cd OpenBench
pip install -e .
cd ..

echo.
echo Installation completed!
echo Run: python -m openbench.openbench --help
pause
```

## 六、验证安装

创建测试脚本 `test_installation.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test OpenBench installation"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    packages = [
        'numpy',
        'pandas',
        'xarray',
        'netCDF4',
        'scipy',
        'matplotlib',
        'cartopy',
        'joblib',
        'dask',
        'jinja2',
    ]

    print("Testing package imports...")
    failed = []

    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError as e:
            print(f"  ✗ {pkg}: {e}")
            failed.append(pkg)

    if failed:
        print(f"\nFailed to import: {', '.join(failed)}")
        return False

    print("\nAll packages imported successfully!")
    return True

def test_openbench():
    """Test if OpenBench can be imported"""
    print("\nTesting OpenBench...")
    try:
        import openbench
        print(f"  ✓ OpenBench version: {openbench.__version__ if hasattr(openbench, '__version__') else 'unknown'}")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import OpenBench: {e}")
        return False

if __name__ == '__main__':
    success = test_imports() and test_openbench()
    sys.exit(0 if success else 1)
```

## 七、常见问题及解决方案

### Q1: cartopy安装失败

**原因**: 缺少GEOS或PROJ系统库

**解决方案**:
```bash
# Linux
sudo apt-get install libgeos-dev libproj-dev

# macOS
brew install geos proj

# 或使用conda安装(推荐)
conda install -c conda-forge cartopy
```

### Q2: netCDF4安装失败

**原因**: 缺少HDF5或netCDF系统库

**解决方案**:
```bash
# Linux
sudo apt-get install libhdf5-dev libnetcdf-dev

# macOS
brew install hdf5 netcdf

# 或使用conda安装(推荐)
conda install -c conda-forge netCDF4
```

### Q3: 包版本冲突

**解决方案**: 使用conda-pack打包整个环境,确保版本一致

### Q4: 不同CPU架构

**解决方案**: 在目标机器相同架构上构建,或使用Docker

## 八、推荐的部署方案

### 方案A: 小型团队 (推荐: pip download)
- 适合: 5-10人,相同操作系统
- 优点: 简单,快速
- 步骤: 下载wheels → 打包 → 分发

### 方案B: 企业部署 (推荐: conda-pack)
- 适合: 多个服务器,需要环境一致性
- 优点: 包含系统依赖,环境完全一致
- 步骤: 打包conda环境 → 分发 → 解压激活

### 方案C: 高安全环境 (推荐: Docker)
- 适合: 隔离环境,容器化部署
- 优点: 最可靠,完全隔离,易于管理
- 步骤: 构建镜像 → 保存 → 加载 → 运行

## 九、最佳实践

1. **版本记录**: 记录所有包的确切版本
2. **平台标记**: 明确标注操作系统和CPU架构
3. **测试验证**: 在目标环境测试后再大规模部署
4. **文档齐全**: 提供详细的离线安装文档
5. **备份方案**: 准备多种安装方法以应对不同情况
6. **更新策略**: 定期更新离线包,修复安全问题

## 十、快速开始 - 制作离线包

**一行命令制作基础离线包**:

```bash
# 创建目录结构
mkdir -p openbench-offline/{packages,OpenBench} && \

# 下载所有依赖
pip download -r requirements.txt -d openbench-offline/packages && \

# 复制源码
cp -r openbench/ setup.py requirements.txt README.md openbench-offline/OpenBench/ && \

# 打包
tar -czf openbench-offline-$(date +%Y%m%d).tar.gz openbench-offline/

echo "Offline package created: openbench-offline-$(date +%Y%m%d).tar.gz"
```

---

**更新时间**: 2025-10-07
**适用版本**: OpenBench v1.0+
**维护者**: OpenBench开发团队
