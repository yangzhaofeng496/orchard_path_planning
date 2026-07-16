"""
局部规划器配置加载器
从 YAML 文件加载 DWA 和 TEB 配置
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from .config import DWAConfig, TEBConfig


def load_yaml(file_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def load_dwa_config(file_path: str) -> DWAConfig:
    """
    从 YAML 文件加载 DWA 配置

    Args:
        file_path: YAML 配置文件路径

    Returns:
        DWAConfig 实例
    """
    config_dict = load_yaml(file_path)

    # 提取 DWA 配置部分
    if 'dwa' not in config_dict:
        raise ValueError("配置文件中缺少 'dwa' 部分")

    dwa_dict = config_dict['dwa']

    # 创建 DWAConfig，只使用配置文件中存在的字段
    return DWAConfig(**{
        k: v for k, v in dwa_dict.items()
        if k in DWAConfig.__dataclass_fields__
    })


def load_teb_config(file_path: str) -> TEBConfig:
    """
    从 YAML 文件加载 TEB 配置

    Args:
        file_path: YAML 配置文件路径

    Returns:
        TEBConfig 实例
    """
    config_dict = load_yaml(file_path)

    # 提取 TEB 配置部分
    if 'teb' not in config_dict:
        raise ValueError("配置文件中缺少 'teb' 部分")

    teb_dict = config_dict['teb']

    # 创建 TEBConfig，只使用配置文件中存在的字段
    return TEBConfig(**{
        k: v for k, v in teb_dict.items()
        if k in TEBConfig.__dataclass_fields__
    })


def save_dwa_config(config: DWAConfig, file_path: str):
    """
    保存 DWA 配置到 YAML 文件

    Args:
        config: DWAConfig 实例
        file_path: YAML 配置文件路径
    """
    from dataclasses import asdict

    config_dict = {'dwa': asdict(config)}

    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)


def save_teb_config(config: TEBConfig, file_path: str):
    """
    保存 TEB 配置到 YAML 文件

    Args:
        config: TEBConfig 实例
        file_path: YAML 配置文件路径
    """
    from dataclasses import asdict

    config_dict = {'teb': asdict(config)}

    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)
