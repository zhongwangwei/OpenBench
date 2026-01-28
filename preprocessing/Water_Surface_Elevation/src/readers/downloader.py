#!/usr/bin/env python3
"""
Unified Data Downloader for WSE Pipeline
统一数据下载管理器

支持的数据源:
- HydroSat: https://hydrosat.gis.uni-stuttgart.de/
- HydroWeb: https://hydroweb.next.theia-land.fr/ (需要账户)
- CGLS: https://land.copernicus.eu/ (需要账户)
- ICESat: https://nsidc.org/data/gla14 (需要 Earthdata 账户)
"""

import os
import re
import glob
import json
import zipfile
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
from abc import ABC, abstractmethod

# SSL configuration - warnings are managed per-instance, not globally
import urllib3
import warnings


class BaseDownloader(ABC):
    """下载器基类"""

    source_name: str = "unknown"
    description: str = ""
    requires_auth: bool = False
    download_url: str = ""

    def __init__(self, output_dir: str, logger=None, verify_ssl: bool = True):
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.verify_ssl = verify_ssl
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Log warning when SSL verification is disabled
        if not self.verify_ssl:
            self.log('warning', f"SSL verification is disabled for {self.source_name}. This is insecure and should only be used for trusted internal networks.")

    def log(self, level: str, message: str):
        if self.logger:
            getattr(self.logger, level.lower())(message)
        else:
            print(f"[{level.upper()}] {message}")

    @abstractmethod
    def check_data_exists(self) -> bool:
        """检查数据是否已存在"""
        pass

    @abstractmethod
    def download(self, **kwargs) -> Path:
        """下载数据"""
        pass

    @abstractmethod
    def get_data_path(self) -> Path:
        """获取数据目录路径"""
        pass

    def ensure_data(self, force_download: bool = False, **kwargs) -> Path:
        """确保数据存在，如果不存在则下载

        Args:
            force_download: 强制下载（即使数据已存在，也会进行增量下载）
        """
        if not force_download and self.check_data_exists():
            data_path = self.get_data_path()
            self.log('info', f"{self.source_name} 数据已存在: {data_path}")
            return data_path
        else:
            if force_download:
                self.log('info', f"{self.source_name} 强制下载/增量更新...")
            else:
                self.log('info', f"{self.source_name} 数据不存在，开始下载...")
            return self.download(**kwargs)


class HydroSatDownloader(BaseDownloader):
    """
    HydroSat 数据下载器

    数据源: https://hydrosat.gis.uni-stuttgart.de/
    直接下载，无需认证
    """

    source_name = "HydroSat"
    description = "HydroSat Water Level (University of Stuttgart)"
    requires_auth = False
    download_url = "https://hydrosat.gis.uni-stuttgart.de/data/download/WL-HydroSat.zip"

    def check_data_exists(self) -> bool:
        data_dir = self.output_dir / "WL_hydrosat"
        if data_dir.exists():
            txt_files = list(data_dir.glob("*.txt"))
            return len(txt_files) > 0
        return False

    def get_data_path(self) -> Path:
        return self.output_dir / "WL_hydrosat"

    def download(self, **kwargs) -> Path:
        zip_path = self.output_dir / "WL-HydroSat.zip"
        extract_dir = self.get_data_path()

        # 下载
        if not zip_path.exists():
            self.log('info', f"下载: {self.download_url}")
            self._download_file(self.download_url, zip_path)

        # 解压
        if not extract_dir.exists():
            self.log('info', f"解压到: {self.output_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.output_dir)

        txt_count = len(list(extract_dir.glob("*.txt")))
        self.log('info', f"下载完成: {txt_count} 个站点文件")

        return extract_dir

    def _download_file(self, url: str, output_path: Path):
        # Suppress InsecureRequestWarning only when verify_ssl is False
        with warnings.catch_warnings():
            if not self.verify_ssl:
                warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning)

            response = requests.get(url, stream=True, timeout=300, verify=self.verify_ssl)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024 * 1024) < 8192:
                        self.log('info', f"下载进度: {downloaded / total_size * 100:.1f}%")

        self.log('info', f"下载完成: {output_path.stat().st_size / (1024*1024):.1f} MB")


class HydroWebDownloader(BaseDownloader):
    """
    HydroWeb 数据下载器

    数据源: https://hydroweb.next.theia-land.fr/
    需要 Theia 账户和 API Key

    获取 API Key:
    1. 注册/登录: https://hydroweb.next.theia-land.fr/
    2. 访问 API 页面: https://hydroweb.next.theia-land.fr/api
    3. 复制页面上的 API Key

    支持四种模式:
    1. 从本地 ZIP 文件解压 (如果已手动下载)
    2. 通过带 token 的 URL 直接下载
    3. 通过 API Key 下载
    4. 交互式输入 API Key 下载
    """

    source_name = "HydroWeb"
    description = "HydroWeb River Altimetry (Theia/CNES)"
    requires_auth = True

    # API 端点
    DOWNLOAD_PAGE = "https://hydroweb.next.theia-land.fr/"
    API_URL = "https://hydroweb.next.theia-land.fr/api"

    # 可用的数据集
    DATASETS = {
        'rivers_research': 'HYDROWEB_RIVERS_RESEARCH',
        'rivers_ope': 'HYDROWEB_RIVERS_OPE',
        'lakes_research': 'HYDROWEB_LAKES_RESEARCH',
        'lakes_ope': 'HYDROWEB_LAKES_OPE',
    }

    # 本地 ZIP 文件名模式
    ZIP_PATTERNS = [
        "Theia_Hydroweb_*River*.zip",
        "hydroweb_river*.zip",
        "HYDROWEB_RIVERS*.zip",
    ]

    def __init__(self, output_dir: str, logger=None, zip_source: str = None, download_url: str = None, verify_ssl: bool = True):
        """
        初始化

        Args:
            output_dir: 输出目录
            logger: 日志记录器
            zip_source: 已下载的 ZIP 文件路径 (可选)
            download_url: 带 token 的下载链接 (可选)
            verify_ssl: 是否验证 SSL 证书 (默认 True)
        """
        super().__init__(output_dir, logger, verify_ssl=verify_ssl)
        self.zip_source = Path(zip_source) if zip_source else None
        self.download_url = download_url

    def check_data_exists(self) -> bool:
        data_dir = self.get_data_path()
        if data_dir.exists():
            txt_files = list(data_dir.glob("*.txt"))
            return len(txt_files) > 0
        return False

    def get_data_path(self) -> Path:
        return self.output_dir / "hydroweb_river"

    def download(self, api_key: str = None, download_url: str = None,
                 interactive: bool = True, datasets: list = None, **kwargs) -> Path:
        """
        下载 HydroWeb 数据

        Args:
            api_key: HydroWeb API Key (从网站账户设置获取)
            download_url: 带 token 的下载链接
            interactive: 是否允许交互式输入 (默认 True)
            datasets: 数据集列表，None 表示下载全部
        """
        extract_dir = self.get_data_path()

        # 方式1: 从指定的 ZIP 文件解压
        if self.zip_source and self.zip_source.exists():
            self.log('info', f"从本地 ZIP 解压: {self.zip_source}")
            return self._extract_zip(self.zip_source, extract_dir)

        # 方式2: 查找本地已有的 ZIP 文件
        zip_file = self._find_local_zip()
        if zip_file:
            self.log('info', f"找到本地 ZIP: {zip_file}")
            return self._extract_zip(zip_file, extract_dir)

        # 方式3: 通过带 token 的 URL 直接下载
        url = download_url or self.download_url
        if url:
            self.log('info', f"通过 token URL 下载...")
            return self._download_via_url(url, extract_dir)

        # 方式4: 从环境变量获取 API Key
        if not api_key:
            api_key = os.environ.get('HYDROWEB_API_KEY')

        # 方式5: 通过 API Key 下载
        if api_key:
            self.log('info', "通过 API Key 下载...")
            return self._download_via_api(api_key, datasets)

        # 方式6: 交互式输入 API Key
        if interactive:
            self.log('info', "HydroWeb 数据需要 API Key")
            print("\n" + "=" * 60)
            print("HydroWeb 数据下载需要 API Key")
            print(f"获取 API Key 步骤:")
            print(f"  1. 注册/登录: {self.DOWNLOAD_PAGE}")
            print(f"  2. 访问 API 页面: {self.API_URL}")
            print(f"  3. 复制页面上的 API Key")
            print("=" * 60)

            try:
                api_key = input("\n请输入 HydroWeb API Key: ").strip()
                if not api_key:
                    raise ValueError("API Key 不能为空")

                # 选择数据集
                print("\n可用数据集:")
                print("  1. rivers_research - HydroWeb Rivers Research")
                print("  2. rivers_ope      - HydroWeb Rivers Operational")
                print("  3. lakes_research  - HydroWeb Lakes Research")
                print("  4. lakes_ope       - HydroWeb Lakes Operational")
                print("  5. all             - 下载全部 (默认)")
                choice = input("\n选择数据集 [1-5, 默认 5]: ").strip() or "5"

                dataset_map = {
                    '1': ['rivers_research'],
                    '2': ['rivers_ope'],
                    '3': ['lakes_research'],
                    '4': ['lakes_ope'],
                    '5': None,  # None 表示全部
                }
                datasets = dataset_map.get(choice, None)

                self.log('info', f"通过 API Key 下载...")
                return self._download_via_api(api_key, datasets)

            except (KeyboardInterrupt, EOFError):
                print("\n已取消")
                raise RuntimeError("用户取消下载")

        # 无法下载
        raise RuntimeError(
            f"无法下载 HydroWeb 数据。请选择以下方式之一:\n"
            f"1. 手动下载 ZIP 文件并放置在 {self.output_dir}\n"
            f"2. 提供带 token 的下载链接\n"
            f"3. 提供 API Key (--api-key 或环境变量 HYDROWEB_API_KEY)\n"
            f"\n获取 API Key: {self.DOWNLOAD_PAGE}"
        )

    def _find_local_zip(self) -> Optional[Path]:
        """查找本地已有的 ZIP 文件"""
        for pattern in self.ZIP_PATTERNS:
            files = list(self.output_dir.glob(pattern))
            if files:
                return files[0]

            # 也检查父目录
            files = list(self.output_dir.parent.glob(pattern))
            if files:
                return files[0]

        return None

    def _extract_zip(self, zip_path: Path, extract_dir: Path) -> Path:
        """解压 ZIP 文件"""
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zf:
            # 获取所有 txt 文件
            txt_files = [f for f in zf.namelist() if f.endswith('.txt')]
            self.log('info', f"解压 {len(txt_files)} 个文件...")

            for txt_file in txt_files:
                # 直接解压到目标目录（忽略 ZIP 内的目录结构）
                filename = Path(txt_file).name
                with zf.open(txt_file) as src:
                    with open(extract_dir / filename, 'wb') as dst:
                        dst.write(src.read())

        self.log('info', f"解压完成: {extract_dir}")
        return extract_dir

    def _download_via_url(self, url: str, extract_dir: Path) -> Path:
        """
        通过带 token 的 URL 下载

        URL 格式:
        https://hydroweb.next.theia-land.fr/download/workflows/{id}/zip?filename={name}&token={jwt}
        """
        # 从 URL 中提取文件名
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        filename = params.get('filename', ['hydroweb_river.zip'])[0]

        zip_path = self.output_dir / filename

        # 下载 ZIP 文件
        if not zip_path.exists():
            self.log('info', f"下载: {filename}")
            self.log('info', f"URL: {url[:80]}...")

            try:
                response = requests.get(
                    url,
                    stream=True,
                    timeout=600,
                    verify=True,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                )
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (1024 * 1024) < 8192:
                            progress = downloaded / total_size * 100
                            self.log('info', f"下载进度: {progress:.1f}% ({downloaded // (1024*1024)} MB)")

                self.log('info', f"下载完成: {zip_path.stat().st_size / (1024*1024):.1f} MB")

            except requests.RequestException as e:
                if zip_path.exists():
                    zip_path.unlink()
                raise RuntimeError(
                    f"下载失败: {e}\n"
                    f"token 可能已过期，请重新获取下载链接:\n"
                    f"  {self.DOWNLOAD_PAGE}"
                )

        # 解压
        return self._extract_zip(zip_path, extract_dir)

    def _download_via_api(self, api_key: str, datasets: list = None) -> Path:
        """
        通过 HydroWeb API 下载

        使用 py-hydroweb 库:
        https://github.com/CNES/py-hydroweb
        """
        try:
            import py_hydroweb
        except ImportError:
            raise RuntimeError(
                "请先安装 py-hydroweb 库:\n"
                "  pip install py-hydroweb"
            )

        extract_dir = self.get_data_path()
        extract_dir.mkdir(parents=True, exist_ok=True)

        # 默认下载所有数据集
        if datasets is None:
            datasets = list(self.DATASETS.keys())

        # 设置 API Key
        os.environ['HYDROWEB_API_KEY'] = api_key

        # 创建客户端
        client = py_hydroweb.Client(api_key=api_key)

        # 为每个数据集下载
        for dataset in datasets:
            collection_id = self.DATASETS.get(dataset, dataset)
            self.log('info', f"下载数据集: {collection_id}")

            zip_path = self.output_dir / f"Theia_Hydroweb_{dataset}.zip"

            # 如果 ZIP 已存在则直接解压
            if zip_path.exists() and zip_path.stat().st_size > 1000:
                self.log('info', f"ZIP 已存在: {zip_path}")
                self._extract_zip(zip_path, extract_dir)
                continue

            try:
                # 创建下载篮
                basket = py_hydroweb.DownloadBasket(download_name=f'wse_pipeline_{dataset}')
                basket.add_collection(collection_id=collection_id)

                self.log('info', f"提交下载请求...")

                # 提交并下载
                downloaded_path = client.submit_and_download_zip(
                    download_basket=basket,
                    zip_filename=zip_path.name,
                    output_folder=str(self.output_dir)
                )

                self.log('info', f"下载完成: {downloaded_path}")

                # 解压
                if Path(downloaded_path).exists():
                    self._extract_zip(Path(downloaded_path), extract_dir)

            except Exception as e:
                self.log('error', f"下载 {dataset} 失败: {e}")
                continue

        # 检查是否有数据
        txt_files = list(extract_dir.glob("*.txt"))
        if not txt_files:
            raise RuntimeError(
                f"下载失败，未获取到数据文件。\n"
                f"请尝试手动下载: {self.DOWNLOAD_PAGE}"
            )

        self.log('info', f"共下载 {len(txt_files)} 个站点文件")
        return extract_dir


class CGLSDownloader(BaseDownloader):
    """
    CGLS (Copernicus Global Land Service) 数据下载器

    数据源: Copernicus Data Space Ecosystem (CDSE)
    产品: River Water Level 2002-present (vector), global, Near Real Time – version 2
    数据格式: GeoJSON

    下载方式:
    1. 从本地已有数据读取
    2. 从 CSV 索引文件获取文件列表，通过 OData API 下载 (需要 CDSE 账户)

    注册账户: https://dataspace.copernicus.eu/
    """

    source_name = "CGLS"
    description = "Copernicus Global Land Service River Water Level"
    requires_auth = True

    DOWNLOAD_PAGE = "https://land.copernicus.eu/en/products/water-bodies/water-level-rivers-near-real-time-v2.0"
    REGISTER_URL = "https://dataspace.copernicus.eu/"

    # CSV 索引文件 (包含所有 GeoJSON 文件路径)
    CSV_INDEX_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/CatalogueCSV/bio-geophysical/river_and_lake_water_level/wl-rivers_global_vector_daily_v2/wl-rivers_global_vector_daily_v2_geojson.csv"

    # OData API
    CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"
    TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    # 数据集标识符
    DATASET_ID = "wl-rivers_global_vector_daily_v2"

    def __init__(self, output_dir: str, logger=None, data_source: str = None, verify_ssl: bool = True):
        """
        初始化

        Args:
            output_dir: 输出目录
            logger: 日志记录器
            data_source: 已有数据目录 (可选)
            verify_ssl: 是否验证 SSL 证书 (默认 True)
        """
        super().__init__(output_dir, logger, verify_ssl=verify_ssl)
        self.data_source = Path(data_source) if data_source else None

    def check_data_exists(self) -> bool:
        # 检查指定的数据源目录
        if self.data_source and self.data_source.exists():
            json_files = list(self.data_source.glob("*.json")) + list(self.data_source.glob("*.geojson"))
            return len(json_files) > 0

        # 检查默认目录
        data_dir = self.get_data_path()
        if data_dir.exists():
            json_files = list(data_dir.glob("*.json")) + list(data_dir.glob("*.geojson"))
            return len(json_files) > 0

        return False

    def get_data_path(self) -> Path:
        if self.data_source and self.data_source.exists():
            return self.data_source
        return self.output_dir / "cgls_river"

    def download(self, username: str = None, password: str = None,
                 interactive: bool = True, max_stations: int = None, **kwargs) -> Path:
        """
        下载 CGLS River Water Level 数据

        Args:
            username: CDSE 账户用户名 (邮箱)
            password: CDSE 账户密码
            interactive: 是否允许交互式输入
            max_stations: 最大下载站点数 (None=全部)
        """
        extract_dir = self.get_data_path()
        extract_dir.mkdir(parents=True, exist_ok=True)

        # 从环境变量获取凭证
        if not username:
            username = os.environ.get('CDSE_USERNAME')
        if not password:
            password = os.environ.get('CDSE_PASSWORD')

        # 交互式输入
        if interactive and (not username or not password):
            self.log('info', "CGLS 数据需要 Copernicus Data Space Ecosystem 账户")
            print("\n" + "=" * 60)
            print("CGLS River Water Level 数据下载")
            print(f"需要 CDSE 账户，免费注册: {self.REGISTER_URL}")
            print("=" * 60)

            try:
                import getpass
                if not username:
                    username = input("\n请输入 CDSE 用户名 (邮箱): ").strip()
                if not password:
                    password = getpass.getpass("请输入 CDSE 密码: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n已取消")
                raise RuntimeError("用户取消下载")

        if not username or not password:
            raise RuntimeError(
                f"CGLS 数据需要 CDSE 账户。\n"
                f"请先注册: {self.REGISTER_URL}\n"
                f"或设置环境变量: CDSE_USERNAME, CDSE_PASSWORD"
            )

        # 获取访问令牌
        self.log('info', "获取访问令牌...")
        token = self._get_access_token(username, password)
        if not token:
            raise RuntimeError("认证失败，请检查用户名和密码")

        self.log('info', "认证成功")

        # 下载数据
        return self._download_via_odata(token, extract_dir, max_stations)

    def _get_access_token(self, username: str, password: str) -> Optional[str]:
        """获取 CDSE OAuth2 访问令牌"""
        try:
            response = requests.post(
                self.TOKEN_URL,
                data={
                    'grant_type': 'password',
                    'username': username,
                    'password': password,
                    'client_id': 'cdse-public',
                },
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return data.get('access_token')

            self.log('warning', f"认证失败: {response.status_code} - {response.text[:200]}")
            return None

        except requests.RequestException as e:
            self.log('error', f"认证请求失败: {e}")
            return None

    def _download_via_odata(self, token: str, extract_dir: Path, max_stations: int = None) -> Path:
        """通过 OData API 下载"""
        import csv
        from io import StringIO

        headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json'
        }

        # Step 1: 下载 CSV 索引获取站点列表
        self.log('info', "下载站点索引...")
        try:
            response = requests.get(self.CSV_INDEX_URL, timeout=120)
            response.raise_for_status()
            csv_content = response.text
        except requests.RequestException as e:
            raise RuntimeError(f"下载索引失败: {e}")

        # Step 2: 解析 CSV 获取唯一站点
        self.log('info', "解析站点列表...")
        reader = csv.DictReader(StringIO(csv_content), delimiter=';')

        # 按站点ID分组，获取最新文件
        stations = {}  # station_id -> latest_record
        for row in reader:
            name = row.get('name', '')
            # 提取站点ID: c_gls_WL_202601170444_0000000005914_ALTI_V2.2.1_geojson
            # 站点ID是第4个部分 (0000000005914)
            parts = name.split('_')
            if len(parts) >= 5:
                station_id = parts[3]  # 0000000005914
                nominal_date = row.get('nominal_date', '')

                if station_id not in stations or nominal_date > stations[station_id]['nominal_date']:
                    stations[station_id] = row

        self.log('info', f"共 {len(stations)} 个站点")

        if max_stations:
            station_list = list(stations.values())[:max_stations]
            self.log('info', f"限制下载前 {max_stations} 个站点")
        else:
            station_list = list(stations.values())

        # Step 3: 下载每个站点的最新 GeoJSON
        self.log('info', f"开始下载 {len(station_list)} 个站点...")
        success_count = 0
        skipped_count = 0

        for i, record in enumerate(station_list):
            product_id = record.get('id')
            name = record.get('name', 'unknown')

            # 检查文件是否已存在
            output_file = extract_dir / f"{name}.geojson"
            if output_file.exists() and output_file.stat().st_size > 100:
                skipped_count += 1
                success_count += 1
                continue

            try:
                # OData 下载 (使用 download.dataspace.copernicus.eu)
                download_url = f"{self.DOWNLOAD_URL}({product_id})/$value"
                response = requests.get(
                    download_url,
                    headers=headers,
                    timeout=60,
                    allow_redirects=True
                )

                if response.status_code == 200:
                    # 返回的是 ZIP 文件，需要解压
                    content = response.content

                    # 检查是否是 ZIP (PK 开头)
                    if content[:2] == b'PK':
                        # 解压 ZIP 提取 GeoJSON
                        import io
                        with zipfile.ZipFile(io.BytesIO(content), 'r') as zf:
                            for zname in zf.namelist():
                                if zname.endswith('.json') or zname.endswith('.geojson'):
                                    json_content = zf.read(zname)
                                    with open(output_file, 'wb') as f:
                                        f.write(json_content)
                                    break
                    else:
                        # 直接保存 GeoJSON
                        with open(output_file, 'wb') as f:
                            f.write(content)

                    success_count += 1
                else:
                    self.log('warning', f"下载失败 {name}: {response.status_code}")

            except Exception as e:
                self.log('warning', f"下载 {name} 失败: {e}")

            if (i + 1) % 100 == 0:
                self.log('info', f"进度: {i + 1}/{len(station_list)} ({success_count} 成功, {skipped_count} 跳过)")

        self.log('info', f"下载完成: {success_count}/{len(station_list)} 个站点 (跳过已存在: {skipped_count})")
        return extract_dir


class ICESatDownloader(BaseDownloader):
    """
    ICESat 数据下载器

    支持数据集:
    - GLAH14: ICESat-1 Land Surface Altimetry (2003-2009, 已退役)
    - ATL13: ICESat-2 Inland Water Surface Height (2018-present)

    数据源: NASA NSIDC
    需要 NASA Earthdata 账户 (免费注册)

    下载方式:
    1. 使用 earthaccess 库 (推荐)
    2. 直接通过 requests 下载 (备用)
    3. 提供已下载文件的 URL 列表

    pip install earthaccess
    """

    source_name = "ICESat"
    description = "ICESat/ICESat-2 Water Surface Elevation (NASA)"
    requires_auth = True

    REGISTER_URL = "https://urs.earthdata.nasa.gov"

    # 数据下载基础 URL
    NSIDC_BASE_URL = "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected"

    # 支持的数据集
    DATASETS = {
        'glah14': {
            'short_name': 'GLAH14',
            'description': 'ICESat-1 Land Surface Altimetry (2003-2009)',
            'temporal': ('2003-02-20', '2009-10-11'),
            'url': 'https://nsidc.org/data/glah14',
            'base_path': '/GLAS/GLAH14',
        },
        'atl13': {
            'short_name': 'ATL13',
            'description': 'ICESat-2 Inland Water Surface Height (2018-present)',
            'temporal': ('2018-10-14', None),  # None = present
            'url': 'https://nsidc.org/data/atl13',
            'base_path': '/ATLAS/ATL13',
        },
    }

    def __init__(self, output_dir: str, logger=None, data_source: str = None, verify_ssl: bool = True):
        super().__init__(output_dir, logger, verify_ssl=verify_ssl)
        self.data_source = Path(data_source) if data_source else None

    def check_data_exists(self) -> bool:
        if self.data_source and self.data_source.exists():
            files = list(self.data_source.glob("*.txt")) + \
                    list(self.data_source.glob("*.h5")) + \
                    list(self.data_source.glob("*.nc"))
            return len(files) > 0

        data_dir = self.get_data_path()
        if data_dir.exists():
            files = list(data_dir.glob("*.txt")) + \
                    list(data_dir.glob("*.h5")) + \
                    list(data_dir.glob("*.nc"))
            return len(files) > 0

        return False

    def get_data_path(self) -> Path:
        if self.data_source and self.data_source.exists():
            return self.data_source
        return self.output_dir / "icesat"

    def download(self, username: str = None, password: str = None,
                 interactive: bool = True, dataset: str = 'atl13',
                 bbox: tuple = None, temporal: tuple = None,
                 max_results: int = None, **kwargs) -> Path:
        """
        下载 ICESat 数据

        Args:
            username: Earthdata 用户名
            password: Earthdata 密码
            interactive: 是否允许交互式输入
            dataset: 数据集 (glah14 或 atl13)
            bbox: 边界框 (west, south, east, north)
            temporal: 时间范围 (start_date, end_date)
            max_results: 最大结果数
        """
        try:
            import earthaccess
        except ImportError:
            raise RuntimeError(
                "请先安装 earthaccess 库:\n"
                "  pip install earthaccess"
            )

        extract_dir = self.get_data_path()
        extract_dir.mkdir(parents=True, exist_ok=True)

        # 获取数据集信息
        if dataset not in self.DATASETS:
            raise ValueError(f"未知数据集: {dataset}。支持: {list(self.DATASETS.keys())}")

        ds_info = self.DATASETS[dataset]
        self.log('info', f"数据集: {ds_info['description']}")

        # 交互式输入
        if interactive and (not username or not password):
            self.log('info', "ICESat 数据需要 NASA Earthdata 账户")
            print("\n" + "=" * 60)
            print("ICESat 数据下载")
            print(f"需要 Earthdata 账户，免费注册: {self.REGISTER_URL}")
            print("=" * 60)

            print("\n可用数据集:")
            print("  1. atl13  - ICESat-2 Inland Water (2018-至今, 推荐)")
            print("  2. glah14 - ICESat-1 Land Surface (2003-2009, 历史)")
            choice = input("\n选择数据集 [1-2, 默认 1]: ").strip() or "1"
            dataset = 'atl13' if choice == '1' else 'glah14'
            ds_info = self.DATASETS[dataset]

            try:
                if not username:
                    username = input("\n请输入 Earthdata 用户名: ").strip()
                if not password:
                    import getpass
                    password = getpass.getpass("请输入 Earthdata 密码: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n已取消")
                raise RuntimeError("用户取消下载")

        # 验证 Earthdata 凭证 (使用 Basic Auth，快速验证)
        self.log('info', "验证 NASA Earthdata 凭证...")

        session = requests.Session()
        session.auth = (username, password)

        try:
            test_resp = session.get(
                f"{self.REGISTER_URL}/users/{username}",
                timeout=30
            )
            if test_resp.status_code != 200:
                if 'locked' in test_resp.text.lower():
                    raise RuntimeError(
                        f"Earthdata 账户已锁定\n\n"
                        f"由于多次登录失败，账户已被临时锁定。\n"
                        f"请等待 10 分钟后重试。"
                    )
                raise RuntimeError(
                    f"Earthdata 凭证验证失败\n\n"
                    f"请检查:\n"
                    f"1. 访问 {self.REGISTER_URL} 确认可以正常登录\n"
                    f"2. 注意用户名区分大小写\n"
                    f"3. 密码可能包含特殊字符，请确保正确输入"
                )
            self.log('info', "凭证验证成功")
        except requests.RequestException as e:
            raise RuntimeError(f"凭证验证失败: {e}")

        # 设置搜索参数
        search_params = {
            'short_name': ds_info['short_name'],
        }

        # 时间范围
        if temporal:
            search_params['temporal'] = temporal
        elif ds_info['temporal'][1]:
            search_params['temporal'] = ds_info['temporal']

        # 空间范围
        if bbox:
            search_params['bounding_box'] = bbox

        # 结果数量
        if max_results:
            search_params['count'] = max_results

        # 搜索数据 (不需要登录，但 earthaccess 可能会自动尝试登录并输出错误)
        self.log('info', f"搜索 {ds_info['short_name']} 数据...")
        results = []
        data_urls = []

        try:
            import earthaccess
            import sys
            from io import StringIO

            # 临时禁用 earthaccess 的自动登录输出 (可能输出到 stdout 或 stderr)
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()

            try:
                results = earthaccess.search_data(**search_params)
            finally:
                # 恢复输出
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            self.log('info', f"找到 {len(results)} 个数据文件")

            if results:
                # 获取下载链接
                for r in results:
                    links = r.data_links()
                    if links:
                        data_urls.extend(links)

        except Exception as e:
            self.log('warning', f"earthaccess 搜索失败: {e}")

        if not data_urls:
            self.log('warning', "无法通过 earthaccess 获取数据链接")
            self.log('info', f"请手动搜索数据: https://search.earthdata.nasa.gov/search?q={ds_info['short_name']}")
            self.log('info', "然后使用 --url-list 参数提供下载链接")
            return extract_dir

        # 方法1: 直接下载 (优先使用，更可靠，不会导致账户锁定)
        self.log('info', f"开始下载 {len(data_urls)} 个文件到 {extract_dir}...")

        # 获取并行下载参数
        num_workers = kwargs.get('num_workers', 5)
        self.log('info', f"使用 {num_workers} 个并行下载线程 (Basic Auth)...")

        # 过滤出需要下载的文件
        urls_to_download = []
        skipped_count = 0
        for url in data_urls:
            filename = url.split('/')[-1]
            filepath = extract_dir / filename
            if filepath.exists() and filepath.stat().st_size > 1000:
                skipped_count += 1
            else:
                urls_to_download.append(url)

        if skipped_count > 0:
            self.log('info', f"跳过已存在的 {skipped_count} 个文件")

        self.log('info', f"需要下载 {len(urls_to_download)} 个文件...")

        # 并行下载函数
        def download_file(args):
            idx, url = args
            filename = url.split('/')[-1]
            filepath = extract_dir / filename

            try:
                # 每个线程使用自己的 session
                thread_session = requests.Session()
                thread_session.auth = (username, password)

                resp = thread_session.get(url, stream=True, timeout=300, allow_redirects=True)

                if resp.status_code == 200:
                    total_size = int(resp.headers.get('content-length', 0))

                    with open(filepath, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)

                    size_mb = filepath.stat().st_size / (1024*1024)
                    return (True, filename, size_mb, None)
                elif resp.status_code == 401:
                    return (False, filename, 0, "认证失败")
                else:
                    return (False, filename, 0, f"HTTP {resp.status_code}")

            except Exception as e:
                return (False, filename, 0, str(e))

        # 使用线程池并行下载
        from concurrent.futures import ThreadPoolExecutor, as_completed

        success_count = skipped_count
        failed_urls = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有下载任务
            future_to_url = {
                executor.submit(download_file, (i, url)): url
                for i, url in enumerate(urls_to_download)
            }

            # 处理完成的任务
            completed = 0
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                completed += 1

                try:
                    success, filename, size_mb, error = future.result()
                    if success:
                        success_count += 1
                        self.log('info', f"[{completed}/{len(urls_to_download)}] ✓ {filename} ({size_mb:.1f} MB)")
                    else:
                        failed_urls.append(url)
                        self.log('warning', f"[{completed}/{len(urls_to_download)}] ✗ {filename}: {error}")
                except Exception as e:
                    failed_urls.append(url)
                    self.log('warning', f"[{completed}/{len(urls_to_download)}] ✗ 下载失败: {e}")

                # 每 50 个文件报告一次总进度
                if completed % 50 == 0:
                    total_done = skipped_count + completed
                    self.log('info', f"总进度: {total_done}/{len(data_urls)} ({total_done/len(data_urls)*100:.1f}%)")

        # 方法2: 如果直接下载失败，尝试使用 earthaccess (最后手段)
        if failed_urls and len(failed_urls) == len(data_urls):
            self.log('info', "直接下载全部失败，尝试使用 earthaccess 下载...")

            try:
                import earthaccess

                # 设置环境变量
                os.environ['EARTHDATA_USERNAME'] = username
                os.environ['EARTHDATA_PASSWORD'] = password

                # 尝试登录
                try:
                    auth = earthaccess.login(strategy='environment')
                    self.log('info', "earthaccess 登录成功")

                    # 使用 earthaccess 下载
                    if results:
                        files = earthaccess.download(results, str(extract_dir))
                        success_count = len(files)
                        self.log('info', f"earthaccess 下载完成: {success_count} 个文件")

                except Exception as e:
                    self.log('error', f"earthaccess 下载也失败: {e}")
                    self.log('info', "建议: 请检查凭证是否正确，或等待账户解锁后重试")

            except ImportError:
                self.log('error', "earthaccess 未安装，无法使用备用下载方式")

        self.log('info', f"下载完成: {success_count}/{len(data_urls)} 个文件")
        return extract_dir

    def download_urls(self, urls: List[str], username: str = None, password: str = None,
                      interactive: bool = True) -> Path:
        """
        直接下载指定的 URL 列表

        当 earthaccess 认证失败时使用此方法

        Args:
            urls: 要下载的文件 URL 列表
            username: Earthdata 用户名
            password: Earthdata 密码
            interactive: 是否允许交互式输入

        Returns:
            下载目录路径
        """
        extract_dir = self.get_data_path()
        extract_dir.mkdir(parents=True, exist_ok=True)

        # 获取凭证
        if not username:
            username = os.environ.get('EARTHDATA_USERNAME')
        if not password:
            password = os.environ.get('EARTHDATA_PASSWORD')

        if interactive and (not username or not password):
            try:
                if not username:
                    username = input("请输入 Earthdata 用户名: ").strip()
                if not password:
                    import getpass
                    password = getpass.getpass("请输入 Earthdata 密码: ").strip()
            except (KeyboardInterrupt, EOFError):
                raise RuntimeError("用户取消下载")

        if not username or not password:
            raise RuntimeError(f"需要 Earthdata 凭证。注册: {self.REGISTER_URL}")

        # 创建带认证的 session
        session = requests.Session()
        session.auth = (username, password)

        # 下载每个文件
        self.log('info', f"开始下载 {len(urls)} 个文件...")
        success_count = 0

        for i, url in enumerate(urls):
            filename = url.split('/')[-1]
            filepath = extract_dir / filename

            if filepath.exists():
                self.log('info', f"跳过已存在: {filename}")
                success_count += 1
                continue

            try:
                self.log('info', f"下载 [{i+1}/{len(urls)}]: {filename}")
                resp = session.get(url, stream=True, timeout=300, allow_redirects=True)

                if resp.status_code == 200:
                    total_size = int(resp.headers.get('content-length', 0))
                    downloaded = 0

                    with open(filepath, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0 and downloaded % (5 * 1024 * 1024) < 8192:
                                self.log('info', f"  进度: {downloaded / total_size * 100:.0f}%")

                    success_count += 1
                    self.log('info', f"  完成: {filepath.stat().st_size / (1024*1024):.1f} MB")

                elif resp.status_code == 401:
                    self.log('error', f"认证失败: {resp.text[:200]}")
                    if 'locked' in resp.text.lower():
                        raise RuntimeError("账户被锁定，请等待 10 分钟后重试")
                    break
                else:
                    self.log('warning', f"下载失败 {filename}: HTTP {resp.status_code}")

            except requests.RequestException as e:
                self.log('warning', f"下载 {filename} 失败: {e}")

        self.log('info', f"下载完成: {success_count}/{len(urls)} 个文件")
        return extract_dir


class DataDownloadManager:
    """
    数据下载管理器

    统一管理所有数据源的下载和检测
    """

    DOWNLOADERS = {
        'hydrosat': HydroSatDownloader,
        'hydroweb': HydroWebDownloader,
        'cgls': CGLSDownloader,
        'icesat': ICESatDownloader,
    }

    def __init__(self, config: Dict[str, Any], logger=None):
        """
        初始化

        Args:
            config: 配置字典 (包含 global_paths)
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        self.downloaders = {}

        self._init_downloaders()

    def _init_downloaders(self):
        """初始化各数据源的下载器"""
        global_paths = self.config.get('global_paths', {})
        data_sources = global_paths.get('data_sources', {})

        # Get SSL verification setting from config (default: True for security)
        verify_ssl = self.config.get('verify_ssl', True)

        for source_name, downloader_cls in self.DOWNLOADERS.items():
            source_config = data_sources.get(source_name, {})
            output_dir = source_config.get('root', f'./data/{source_name}')

            # 特殊参数
            kwargs = {}
            if source_name == 'hydroweb':
                kwargs['zip_source'] = source_config.get('zip_file')
            elif source_name in ['cgls', 'icesat']:
                kwargs['data_source'] = source_config.get('root')

            self.downloaders[source_name] = downloader_cls(
                output_dir=output_dir,
                logger=self.logger,
                verify_ssl=verify_ssl,
                **kwargs
            )

    def check_all(self) -> Dict[str, bool]:
        """检查所有数据源的状态"""
        status = {}
        for name, downloader in self.downloaders.items():
            status[name] = downloader.check_data_exists()
        return status

    def ensure_data(self, source: str, **kwargs) -> Path:
        """
        确保指定数据源的数据存在

        Args:
            source: 数据源名称
            **kwargs: 额外参数 (如认证信息)

        Returns:
            数据目录路径
        """
        if source not in self.downloaders:
            raise ValueError(f"未知的数据源: {source}")

        return self.downloaders[source].ensure_data(**kwargs)

    def get_data_path(self, source: str) -> Path:
        """获取数据路径"""
        if source not in self.downloaders:
            raise ValueError(f"未知的数据源: {source}")

        return self.downloaders[source].get_data_path()

    def print_status(self):
        """打印所有数据源状态"""
        print("\n数据源状态:")
        print("-" * 60)

        for name, downloader in self.downloaders.items():
            exists = downloader.check_data_exists()
            status = "✓ 已就绪" if exists else "✗ 未下载"
            auth = "(需要认证)" if downloader.requires_auth else "(免费下载)"

            print(f"  {name:12} {status:12} {auth}")

            if exists:
                path = downloader.get_data_path()
                print(f"               路径: {path}")
            else:
                print(f"               下载: {downloader.download_url or downloader.DOWNLOAD_PAGE}")

        print("-" * 60)


def get_downloader(source: str, output_dir: str, logger=None, **kwargs) -> BaseDownloader:
    """
    获取指定数据源的下载器

    Args:
        source: 数据源名称 (hydrosat, hydroweb, cgls, icesat)
        output_dir: 输出目录
        logger: 日志记录器
        **kwargs: 额外参数

    Returns:
        下载器实例
    """
    downloaders = {
        'hydrosat': HydroSatDownloader,
        'hydroweb': HydroWebDownloader,
        'cgls': CGLSDownloader,
        'icesat': ICESatDownloader,
    }

    if source not in downloaders:
        raise ValueError(f"未知的数据源: {source}。支持: {list(downloaders.keys())}")

    return downloaders[source](output_dir, logger, **kwargs)


# 命令行接口
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='WSE Pipeline 数据下载管理器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据源:
  hydrosat  - HydroSat (免费下载)
  hydroweb  - HydroWeb (需要 Theia 账户)
  cgls      - CGLS Copernicus (需要账户)
  icesat    - ICESat GLA14 (需要 Earthdata 账户)

示例:
  # 检查所有数据源状态
  python -m src.readers.downloader --status

  # 下载 HydroSat 数据 (免费，无需认证)
  python -m src.readers.downloader --download hydrosat --output /path/to/data

  # 从本地 ZIP 解压 HydroWeb 数据
  python -m src.readers.downloader --download hydroweb --output /path/to/data \\
      --zip /path/to/Theia_Hydroweb_River.zip

  # 通过带 token 的 URL 下载 HydroWeb 数据
  python -m src.readers.downloader --download hydroweb --output /path/to/data \\
      --url "https://hydroweb.next.theia-land.fr/download/workflows/.../zip?token=..."

  # 使用 API Key 下载 HydroWeb 数据
  python -m src.readers.downloader --download hydroweb --output /path/to/data \\
      --api-key YOUR_API_KEY

  # 使用环境变量设置 API Key
  export HYDROWEB_API_KEY=YOUR_API_KEY
  python -m src.readers.downloader --download hydroweb --output /path/to/data

  # 交互式下载 HydroWeb (会提示输入 API Key)
  python -m src.readers.downloader --download hydroweb --output /path/to/data

  # 下载 CGLS River Water Level (需要 CDSE 账户，免费注册)
  python -m src.readers.downloader --download cgls --output /path/to/data \\
      --username your@email.com --password yourpass

  # 下载 CGLS 前 100 个站点 (测试)
  python -m src.readers.downloader --download cgls --output /path/to/data --max-stations 100

  # 交互式下载 CGLS
  python -m src.readers.downloader --download cgls --output /path/to/data

  # 下载 ICESat-2 ATL13 数据 (需要 Earthdata 账户，免费注册)
  python -m src.readers.downloader --download icesat --output /path/to/data \\
      --username your_username --password yourpass

  # 下载指定区域和时间的 ICESat 数据
  python -m src.readers.downloader --download icesat --output /path/to/data \\
      --bbox -10,20,10,50 --temporal 2020-01-01,2020-12-31 --max-results 10

  # 下载 ICESat-1 GLAH14 历史数据
  python -m src.readers.downloader --download icesat --output /path/to/data \\
      --icesat-dataset glah14

  # 直接通过 URL 下载 ICESat 数据 (当 earthaccess 认证失败时)
  python -m src.readers.downloader --download icesat --output /path/to/data \\
      --url-list "https://data.nsidc.earthdatacloud.nasa.gov/.../ATL13_xxx.h5" \\
      --username your_username --password yourpass

  # 从 URL 文件批量下载
  python -m src.readers.downloader --download icesat --output /path/to/data \\
      --urls /path/to/url_list.txt --username your_username --password yourpass
        """
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='显示所有数据源状态'
    )

    parser.add_argument(
        '--download',
        metavar='SOURCE',
        choices=['hydrosat', 'hydroweb', 'cgls', 'icesat'],
        help='下载指定数据源'
    )

    parser.add_argument(
        '--output', '-o',
        metavar='DIR',
        default='./data',
        help='输出目录 (默认: ./data)'
    )

    parser.add_argument(
        '--zip',
        metavar='FILE',
        help='本地 ZIP 文件路径 (用于 HydroWeb)'
    )

    parser.add_argument(
        '--url',
        metavar='URL',
        help='带 token 的下载链接 (用于 HydroWeb)'
    )

    parser.add_argument(
        '--api-key', '-k',
        metavar='KEY',
        help='HydroWeb API Key (或设置环境变量 HYDROWEB_API_KEY)'
    )

    parser.add_argument(
        '--dataset',
        metavar='NAME',
        choices=['rivers_research', 'rivers_ope', 'lakes_research', 'lakes_ope', 'all'],
        default='all',
        help='HydroWeb 数据集 (默认: all 下载全部)'
    )

    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='禁用交互式输入'
    )

    parser.add_argument(
        '--username', '-u',
        metavar='USER',
        help='CDSE 账户用户名 (用于 CGLS)'
    )

    parser.add_argument(
        '--password', '-p',
        metavar='PASS',
        help='CDSE 账户密码 (用于 CGLS)'
    )

    parser.add_argument(
        '--max-stations',
        type=int,
        metavar='N',
        help='最大下载站点数 (用于 CGLS)'
    )

    parser.add_argument(
        '--icesat-dataset',
        choices=['atl13', 'glah14', 'all'],
        default='all',
        help='ICESat 数据集 (默认: all 同时下载 ATL13 和 GLAH14)'
    )

    parser.add_argument(
        '--bbox',
        metavar='W,S,E,N',
        help='边界框 (west,south,east,north)'
    )

    parser.add_argument(
        '--temporal',
        metavar='START,END',
        help='时间范围 (YYYY-MM-DD,YYYY-MM-DD)'
    )

    parser.add_argument(
        '--max-results',
        type=int,
        metavar='N',
        help='最大结果数 (用于 ICESat)'
    )

    parser.add_argument(
        '--num-workers', '-j',
        type=int,
        default=5,
        metavar='N',
        help='并行下载线程数 (默认: 5，用于 ICESat)'
    )

    parser.add_argument(
        '--urls',
        metavar='FILE',
        help='包含下载 URL 的文件 (每行一个 URL，用于 ICESat)'
    )

    parser.add_argument(
        '--url-list',
        nargs='+',
        metavar='URL',
        help='直接指定下载 URL (用于 ICESat)'
    )

    args = parser.parse_args()

    if args.status:
        # 简单状态检查
        print("\n数据源下载器:")
        print("-" * 50)
        for name in ['hydrosat', 'hydroweb', 'cgls', 'icesat']:
            downloader = get_downloader(name, args.output)
            exists = downloader.check_data_exists()
            status = "✓" if exists else "✗"
            auth = "(需要认证)" if downloader.requires_auth else ""
            print(f"  {status} {name:12} {auth}")
        print("-" * 50)

    elif args.download:
        print(f"下载 {args.download} 到 {args.output}")

        kwargs = {}
        download_kwargs = {}

        if args.download == 'hydroweb':
            if args.zip:
                kwargs['zip_source'] = args.zip
            if args.url:
                kwargs['download_url'] = args.url
            # 下载时传递认证参数
            if args.api_key:
                download_kwargs['api_key'] = args.api_key
            # datasets=None 表示下载全部
            if args.dataset and args.dataset != 'all':
                download_kwargs['datasets'] = [args.dataset]
            else:
                download_kwargs['datasets'] = None  # 下载全部
            download_kwargs['interactive'] = not args.no_interactive

        elif args.download == 'cgls':
            # CGLS 需要 CDSE 账户
            if args.username:
                download_kwargs['username'] = args.username
            if args.password:
                download_kwargs['password'] = args.password
            if args.max_stations:
                download_kwargs['max_stations'] = args.max_stations
            download_kwargs['interactive'] = not args.no_interactive

        elif args.download == 'icesat':
            # ICESat 需要 Earthdata 账户
            if args.username:
                download_kwargs['username'] = args.username
            if args.password:
                download_kwargs['password'] = args.password
            if args.bbox:
                # 解析 bbox: west,south,east,north
                bbox = tuple(map(float, args.bbox.split(',')))
                download_kwargs['bbox'] = bbox
            if args.temporal:
                # 解析 temporal: start,end
                temporal = tuple(args.temporal.split(','))
                download_kwargs['temporal'] = temporal
            if args.max_results:
                download_kwargs['max_results'] = args.max_results
            download_kwargs['num_workers'] = args.num_workers
            download_kwargs['interactive'] = not args.no_interactive

            # 检查是否使用 URL 直接下载
            urls_to_download = []
            if args.urls:
                # 从文件读取 URL
                with open(args.urls, 'r') as f:
                    urls_to_download = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            if args.url_list:
                urls_to_download.extend(args.url_list)

            if urls_to_download:
                # 使用 URL 直接下载
                downloader = get_downloader(args.download, args.output, **kwargs)
                try:
                    path = downloader.download_urls(
                        urls_to_download,
                        username=args.username,
                        password=args.password,
                        interactive=not args.no_interactive
                    )
                    print(f"\n完成! 数据位置: {path}")
                except Exception as e:
                    print(f"\n错误: {e}")
                exit(0)

            # ICESat 支持下载多个数据集
            if args.icesat_dataset == 'all':
                # 下载所有数据集 (ATL13 和 GLAH14)
                downloader = get_downloader(args.download, args.output, **kwargs)
                datasets_to_download = ['atl13', 'glah14']
                for ds in datasets_to_download:
                    print(f"\n=== 下载 ICESat {ds.upper()} ===")
                    download_kwargs['dataset'] = ds
                    try:
                        path = downloader.download(**download_kwargs)
                        print(f"完成! {ds.upper()} 数据位置: {path}")
                    except Exception as e:
                        print(f"错误 ({ds}): {e}")
                exit(0)
            else:
                download_kwargs['dataset'] = args.icesat_dataset

        downloader = get_downloader(args.download, args.output, **kwargs)

        try:
            # 对于 CGLS 和 ICESat，直接调用 download 进行增量下载
            # 对于其他数据源，使用 ensure_data
            if args.download in ['cgls', 'icesat']:
                path = downloader.download(**download_kwargs)
            else:
                path = downloader.ensure_data(**download_kwargs)
            print(f"\n完成! 数据位置: {path}")
        except NotImplementedError as e:
            print(f"\n{e}")
        except Exception as e:
            print(f"\n错误: {e}")

    else:
        parser.print_help()
