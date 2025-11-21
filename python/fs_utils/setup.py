from setuptools import setup, find_packages

setup(
    name="fs_utils",                # パッケージ名（Pythonは小文字が基本）
    version="0.1.0",
    description="Filesystem utilities (make_dirs, etc.)",
    author="Mitsuki ISHIYAMA",
    packages=find_packages(),       # fs_utils/ を自動検出
)
