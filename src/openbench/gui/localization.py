"""Small runtime English/Chinese translator for the Qt GUI."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QTabWidget,
    QTableWidget,
    QTreeWidget,
    QWidget,
)

from openbench.config.user_settings import load_user_settings, save_user_settings

logger = logging.getLogger(__name__)

LANGUAGE_KEY = "gui_language"
ENGLISH = "en"
CHINESE = "zh_CN"


ZH_CN = {
    "OpenBench NML Wizard": "OpenBench NML 配置向导",
    "NML Configuration Wizard": "NML 配置向导",
    "Load Config...": "加载配置...",
    "New Config": "新建配置",
    "Back": "上一步",
    "Next": "下一步",
    "Finish": "完成",
    "Run": "运行",
    "Rerun": "重新运行",
    "General": "常规",
    "General Settings": "常规设置",
    "Configure basic project settings and evaluation options": "配置项目基本信息和评估选项",
    "Data Registry": "数据注册表",
    "Browse and edit registered model profiles and reference datasets": "浏览和编辑已注册的模型配置与参考数据集",
    "Simulation Data": "模拟数据",
    "Scan a directory for simulation cases, assign models, and select cases to evaluate": "扫描模拟案例目录、分配模型并选择要评估的案例",
    "Reference Data": "参考数据",
    "Configure reference data sources for each evaluation variable": "为每个评估变量配置参考数据源",
    "Evaluation": "评估",
    "Evaluation Items": "评估项目",
    "Select the variables to evaluate": "选择要评估的变量",
    "Carbon Cycle": "碳循环",
    "Water Cycle": "水循环",
    "Energy Cycle": "能量循环",
    "Atmospheric": "大气",
    "Agriculture": "农业",
    "Water Bodies": "水体",
    "Urban": "城市",
    "Metrics": "指标",
    "Select evaluation metrics": "选择评估指标",
    "Basic Metrics": "基础指标",
    "Correlation": "相关性",
    "Efficiency": "效率指标",
    "Hydrology": "水文指标",
    "Other": "其他",
    "Scores": "评分",
    "Select scoring methods": "选择评分方法",
    "ILAMB Scoring System": "ILAMB 评分体系",
    "Comparisons": "对比分析",
    "Select comparison visualizations and analyses": "选择对比可视化与分析方法",
    "Diagrams": "图解",
    "Plots": "绘图",
    "Aggregation": "聚合",
    "Statistics": "统计分析",
    "Select statistical analyses": "选择统计分析方法",
    "Basic Statistics": "基础统计",
    "Advanced Statistics": "高级统计",
    "Runtime Environment": "运行环境",
    "Configure where OpenBench will run - locally on this machine or on a remote server.": "配置 OpenBench 在本机或远程服务器上运行。",
    "Preview & Run": "预览并运行",
    "Preview & Export": "预览并导出",
    "Review generated configuration and export files": "检查生成的配置并导出文件",
    "Run Monitor": "运行监控",
    "Run & Monitor": "运行与监控",
    "Monitor evaluation progress": "监控评估进度",
    "Project Information": "项目信息",
    "Project Name:": "项目名称：",
    "Project name (e.g., Initial_test)": "项目名称（例如 Initial_test）",
    "Output Directory:": "输出目录：",
    "Confirm": "确认",
    "Spatial-Temporal Settings": "时空设置",
    "Year Range:": "年份范围：",
    "Min Year Threshold:": "最少年数阈值：",
    "Minimum number of years of valid data required": "有效数据所需的最少年数",
    "Latitude Range:": "纬度范围：",
    "Longitude Range:": "经度范围：",
    "Time Resolution:": "时间分辨率：",
    "Grid Resolution:": "网格分辨率：",
    "Time Alignment:": "时间对齐：",
    "Timezone:": "时区：",
    "Weight:": "权重：",
    "to": "至",
    "Feature Toggles": "功能开关",
    "Comparison": "对比",
    "Debug Mode": "调试模式",
    "Generate Report": "生成报告",
    "Only Drawing": "仅绘图",
    "Unified Mask": "统一掩膜",
    "Groupby Options": "分组选项",
    "IGBP Groupby": "按 IGBP 分组",
    "PFT Groupby": "按 PFT 分组",
    "Climate Zone Groupby": "按气候区分组",
    "Performance Settings": "性能设置",
    "Compress NetCDF outputs": "压缩 NetCDF 输出",
    "Compression Level:": "压缩级别：",
    "Multi-file Combine:": "多文件合并：",
    "Auto Min Files:": "自动合并最少文件数：",
    "Auto Max Batch:": "自动合并最大批次：",
    "Memory Fraction:": "内存占比：",
    "Enable dask.distributed": "启用 dask.distributed",
    "Workers:": "工作进程数：",
    "Threads/Worker:": "每个进程的线程数：",
    "Use processes": "使用多进程",
    "Memory Limit:": "内存限制：",
    "Search...": "搜索...",
    "Select All": "全选",
    "Deselect All": "取消全选",
    "Select None": "全不选",
    "Models": "模型",
    "Model Profile": "模型配置",
    "Reference Dataset": "参考数据集",
    "Reference Datasets": "参考数据集",
    "+ New Model": "+ 新建模型",
    "+ New Dataset": "+ 新建数据集",
    "+ Add Variable": "+ 添加变量",
    "+ Add Fallback": "+ 添加备用变量",
    "Import": "导入",
    "Import Selected": "导入所选项",
    "Delete": "删除",
    "Remove Selected": "移除所选项",
    "Edit Variable...": "编辑变量...",
    "Revert": "还原",
    "Save": "保存",
    "Save As...": "另存为...",
    "Name:": "名称：",
    "Description:": "描述：",
    "Variables:": "变量：",
    "Reference Scan Root": "参考数据扫描根目录",
    "Directory to scan for reference datasets (e.g., /Volumes/work/Reference)": "用于扫描参考数据集的目录（例如 /Volumes/work/Reference）",
    "Browse": "浏览",
    "Browse...": "浏览...",
    "Scan": "扫描",
    "Scan Directory": "扫描目录",
    "Scan for Datasets": "扫描数据集",
    "Scan for Cases": "扫描案例",
    "Validate Data": "验证数据",
    "Dataset:": "数据集：",
    "Manage datasets in Data Registry": "在数据注册表中管理数据集",
    "Manage models in Data Registry": "在数据注册表中管理模型",
    "No evaluation items selected. Please go back and select items.": "尚未选择评估项目，请返回选择。",
    "Optional Overrides for Selected Cases": "所选案例的可选覆盖设置",
    "Execution Mode": "执行模式",
    "Mode:": "模式：",
    "Local": "本地",
    "Local Python Environment": "本地 Python 环境",
    "Remote Python Environment": "远程 Python 环境",
    "Remote Server": "远程服务器",
    "Parallel Processing": "并行处理",
    "CPU Cores:": "CPU 核心数：",
    "Conda Env:": "Conda 环境：",
    "Conda:": "Conda：",
    "Python:": "Python：",
    "Host:": "主机：",
    "Auth:": "认证：",
    "Status:": "状态：",
    "Node:": "节点：",
    "SSH Key": "SSH 密钥",
    "None (internal trust)": "无（内部信任）",
    "Save Settings": "保存设置",
    "Load Settings": "加载设置",
    "Reset": "重置",
    "Detect": "检测",
    "Connect": "连接",
    "Disconnect": "断开连接",
    "Refresh": "刷新",
    "Install": "安装",
    "Path:": "路径：",
    "Root directory:": "根目录：",
    "Password": "密码",
    "Compute Node (Optional)": "计算节点（可选）",
    "Progress": "进度",
    "Resources": "资源",
    "Log Output": "日志输出",
    "Stop": "停止",
    "Open Output Folder": "打开输出目录",
    "Download Folder to Local...": "下载目录到本地...",
    "Copy to Clipboard": "复制到剪贴板",
    "Retry": "重试",
    "Synced": "已同步",
    "Close": "关闭",
    "Cancel": "取消",
    "OK": "确定",
    "&OK": "确定",
    "&Yes": "是",
    "&No": "否",
    "Yes": "是",
    "No": "否",
    "New": "新建",
    "Registered": "已注册",
    "Unregistered": "未注册",
    "Open": "打开",
    "Export": "导出",
    "Select": "选择",
    "Advanced": "高级设置",
    "Register/Update Selected": "注册/更新所选项",
    "Scanning": "正在扫描",
    "Downloading": "正在下载",
    "Preparing...": "正在准备...",
    "Validating Data...": "正在验证数据...",
    "Data Validation Results": "数据验证结果",
    "Reference Datasets Found": "发现参考数据集",
    "Confirm Scan Results": "确认扫描结果",
    "Select cases to run and assign models:": "选择要运行的案例并分配模型：",
    "Variable Mapping": "变量映射",
    "Variable Mappings": "变量映射",
    "Variable name:": "变量名称：",
    "NC varname:": "NC 变量名：",
    "Unit:": "单位：",
    "Compute:": "计算表达式：",
    "Fallback Variables": "备用变量",
    "Edit Variable Mapping": "编辑变量映射",
    "Add Fallback Variable": "添加备用变量",
    "Import Variables from NetCDF": "从 NetCDF 导入变量",
    "NC file:": "NC 文件：",
    "Path to .nc file": ".nc 文件路径",
    "Variable": "变量",
    "Variables": "变量",
    "Dataset": "数据集",
    "Resolution": "分辨率",
    "Type": "类型",
    "Files": "文件",
    "Data Source": "数据源",
    "Status": "状态",
    "Inspecting": "正在检查",
    "Inspecting selected reference datasets...": "正在检查所选参考数据集...",
    "Case": "案例",
    "Path": "路径",
    "Model": "模型",
    "Variable Name in File": "文件中的变量名",
    "Units": "单位",
    "Dimensions": "维度",
    "New Model Definition": "新建模型定义",
    "Model Information": "模型信息",
    "Model Name:": "模型名称：",
    "Define variable names and units for each evaluation variable:": "为每个评估变量定义变量名和单位：",
    "Error": "错误",
    "Success": "成功",
    "Failed": "失败",
    "Complete": "完成",
    "Exit": "退出",
    "No Selection": "未选择",
    "No Data": "无数据",
    "No Path": "未指定路径",
    "Invalid Path": "路径无效",
    "Invalid File": "文件无效",
    "Validation Failed": "验证失败",
    "Validation Error": "验证错误",
    "Scan Failed": "扫描失败",
    "Scan Incomplete": "扫描未完成",
    "Scan Complete": "扫描完成",
    "No supported reference datasets found.": "未发现受支持的参考数据集。",
    "Save Failed": "保存失败",
    "Export Error": "导出错误",
    "Not Connected": "未连接",
    "Not connected": "未连接",
    "Connected": "已连接",
    "Connecting...": "正在连接...",
    "Cancelled": "已取消",
    "Already Running": "已在运行",
    "Evaluation Running": "评估正在运行",
    "Load Configuration": "加载配置",
    "Load Configuration from Remote Server": "从远程服务器加载配置",
    "Save runtime settings to a file": "将运行设置保存到文件",
    "Load runtime settings from a file": "从文件加载运行设置",
    "Clear cached settings and reset to defaults": "清除缓存设置并恢复默认值",
    "Number of CPU cores to use for parallel processing": "用于并行处理的 CPU 核心数",
    "Refresh conda environments": "刷新 Conda 环境",
    "Auto-detect Python interpreters": "自动检测 Python 解释器",
    "Browse for Python interpreter": "浏览并选择 Python 解释器",
    "Path to OpenBench installation directory": "OpenBench 安装目录路径",
    "Browse for OpenBench installation directory": "浏览 OpenBench 安装目录",
    "Install OpenBench from GitHub": "从 GitHub 安装 OpenBench",
    "Click to select or type user@host": "点击选择或输入 user@host",
    "Connect to SSH server": "连接 SSH 服务器",
    "Disconnect from SSH server": "断开 SSH 服务器连接",
    "Save password (encrypted)": "保存密码（加密）",
    "Connect to compute node via SSH": "通过 SSH 连接计算节点",
    "Disconnect from compute node": "断开计算节点连接",
    "Node password": "节点密码",
    "Path to SSH key for compute node": "计算节点 SSH 密钥路径",
    "Refresh conda environments from remote server": "刷新远程服务器的 Conda 环境",
    "Create new OpenBench conda environment": "创建新的 OpenBench Conda 环境",
    "Detect Python interpreters on remote server": "检测远程服务器上的 Python 解释器",
    "Enter Python path on remote server manually": "手动输入远程服务器上的 Python 路径",
    "Browse remote server for OpenBench installation path": "浏览远程服务器上的 OpenBench 安装路径",
    "Install OpenBench on remote server": "在远程服务器上安装 OpenBench",
    "Simulation root directory (e.g. /data/Simulation)": "模拟数据根目录（例如 /data/Simulation）",
    "List subdirectories that contain NetCDF simulation output": "列出包含 NetCDF 模拟输出的子目录",
    "Check that simulation files exist": "检查模拟文件是否存在",
    "Scan the data root directory for reference datasets and register new ones": "扫描参考数据根目录并注册或更新数据集",
    "Check files, variable names, time and spatial ranges for all data sources": "检查所有数据源的文件、变量名以及时间和空间范围",
    "Edit per-variable varname, unit, prefix, and suffix": "编辑每个变量的变量名、单位、前缀和后缀",
}


def translate_text(text: str, language: str = ENGLISH) -> str:
    """Translate a GUI string while leaving user data and unknown text intact."""
    if language != CHINESE or not text:
        return text
    if text in ZH_CN:
        return ZH_CN[text]

    match = re.fullmatch(r"Step (\d+) of (\d+)", text)
    if match:
        return f"第 {match.group(1)} 步，共 {match.group(2)} 步"
    match = re.fullmatch(r"Selected: (\d+)(?: / (\d+))?", text)
    if match:
        suffix = f" / {match.group(2)}" if match.group(2) else ""
        return f"已选择：{match.group(1)}{suffix}"
    match = re.fullmatch(r"Registry: (\d+) datasets available", text)
    if match:
        return f"注册表：可用数据集 {match.group(1)} 个"
    match = re.fullmatch(r"Registered/updated (\d+) dataset\(s\)\.", text)
    if match:
        return f"已注册/更新 {match.group(1)} 个数据集。"
    match = re.fullmatch(
        r"Registered/updated (\d+) dataset\(s\)\.\nThey are now available in the dropdown menus below\.", text
    )
    if match:
        return f"已注册/更新 {match.group(1)} 个数据集。\n现在可在下方的下拉菜单中选择。"
    match = re.fullmatch(
        r"<b>(\d+) dataset group\(s\)</b> found in the reference data directory\.\n"
        r"Select which to register or update in the OpenBench registry\.",
        text,
    )
    if match:
        return f"在参考数据目录中发现 <b>{match.group(1)} 个数据集组</b>。\n请选择要注册或更新的项目。"
    match = re.fullmatch(r"Validation complete: (\d+) passed, (\d+) failed", text)
    if match:
        return f"验证完成：通过 {match.group(1)} 项，失败 {match.group(2)} 项"
    match = re.fullmatch(r"\(Available: (.+)\)", text)
    if match:
        return f"（可用：{match.group(1)}）"
    return text


class LanguageManager(QObject):
    """Persist the selected language and retranslate visible Qt widgets."""

    language_changed = Signal(str)

    def __init__(self, parent=None, user_dir: str | Path | None = None, persist: bool = True):
        super().__init__(parent)
        self._user_dir = user_dir
        self._persist = persist
        saved = load_user_settings(user_dir).get(LANGUAGE_KEY) if persist else ENGLISH
        self.language = saved if saved in {ENGLISH, CHINESE} else ENGLISH

    def set_language(self, language: str) -> None:
        """Set, save, and apply a supported language."""
        language = language if language in {ENGLISH, CHINESE} else ENGLISH
        if language == self.language:
            return
        self.language = language
        if self._persist:
            try:
                settings = load_user_settings(self._user_dir)
                settings[LANGUAGE_KEY] = language
                save_user_settings(settings, self._user_dir)
            except OSError as exc:
                logger.warning("Could not save GUI language preference: %s", exc)
        self.language_changed.emit(language)
        app = QApplication.instance()
        if app:
            for widget in app.topLevelWidgets():
                self.apply(widget)

    def toggle(self) -> None:
        """Switch between English and Simplified Chinese."""
        self.set_language(CHINESE if self.language == ENGLISH else ENGLISH)

    def apply(self, root: QWidget) -> None:
        """Retranslate a widget tree without changing unknown or user-provided text."""
        for widget in [root, *root.findChildren(QWidget)]:
            if not widget.property("i18n_skip"):
                self._translate_widget(widget)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API name
        if (
            event.type() == QEvent.Show
            and isinstance(watched, QWidget)
            and not watched.property("i18n_skip")
        ):
            self._translate_widget(watched)
        return super().eventFilter(watched, event)

    def _translate_widget(self, widget: QWidget) -> None:
        self._translate_property(widget, "window_title", widget.windowTitle, widget.setWindowTitle)
        self._translate_property(widget, "tooltip", widget.toolTip, widget.setToolTip)

        if isinstance(widget, QAbstractButton):
            self._translate_property(widget, "text", widget.text, widget.setText)
        elif isinstance(widget, QLabel):
            self._translate_property(widget, "text", widget.text, widget.setText)
        elif isinstance(widget, QGroupBox):
            self._translate_property(widget, "title", widget.title, widget.setTitle)

        if isinstance(widget, QLineEdit):
            self._translate_property(widget, "placeholder", widget.placeholderText, widget.setPlaceholderText)
        if isinstance(widget, QTabWidget):
            for index in range(widget.count()):
                self._translate_property(
                    widget,
                    f"tab_{index}",
                    lambda index=index: widget.tabText(index),
                    lambda value, index=index: widget.setTabText(index, value),
                )
        if isinstance(widget, QListWidget) and widget.property("i18n_items"):
            for index in range(widget.count()):
                self._translate_item(widget.item(index))
        if isinstance(widget, QTreeWidget) and widget.property("i18n_items"):
            pending = [widget.topLevelItem(index) for index in range(widget.topLevelItemCount())]
            while pending:
                item = pending.pop()
                for column in range(item.columnCount()):
                    self._translate_item(item, column)
                pending.extend(item.child(index) for index in range(item.childCount()))
        if isinstance(widget, (QTableWidget, QTreeWidget)):
            for index in range(widget.columnCount()):
                item = widget.horizontalHeaderItem(index) if isinstance(widget, QTableWidget) else widget.headerItem()
                if item is not None:
                    self._translate_item(item, index if isinstance(widget, QTreeWidget) else 0)

    def _translate_property(self, obj, key: str, getter, setter) -> None:
        current = getter()
        if not current:
            return
        source_key = f"_i18n_source_{key}"
        rendered_key = f"_i18n_rendered_{key}"
        source = obj.property(source_key)
        rendered = obj.property(rendered_key)
        if source is None or (rendered is not None and current != rendered):
            source = current
            obj.setProperty(source_key, source)
        translated = translate_text(source, self.language)
        if current != translated:
            setter(translated)
        obj.setProperty(rendered_key, translated)

    def _translate_item(self, item, column: int = 0) -> None:
        source_role = int(Qt.UserRole) + 100
        rendered_role = source_role + 1
        current = item.text(column) if hasattr(item, "columnCount") else item.text()
        source = item.data(column, source_role) if hasattr(item, "columnCount") else item.data(source_role)
        rendered = item.data(column, rendered_role) if hasattr(item, "columnCount") else item.data(rendered_role)
        if source is None or (rendered is not None and current != rendered):
            source = current
            if hasattr(item, "columnCount"):
                item.setData(column, source_role, source)
            else:
                item.setData(source_role, source)
        translated = translate_text(source, self.language)
        if hasattr(item, "columnCount"):
            item.setText(column, translated)
            item.setData(column, rendered_role, translated)
        else:
            item.setText(translated)
            item.setData(rendered_role, translated)


def get_language_manager() -> LanguageManager:
    """Return the QApplication-wide language manager."""
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("QApplication must exist before creating the language manager")
    manager = getattr(app, "_openbench_language_manager", None)
    if manager is None:
        manager = LanguageManager(app)
        app._openbench_language_manager = manager
        app.installEventFilter(manager)
    return manager
