from setuptools import setup, find_packages

setup(
    name="image_utils",             # PyPI的に小文字が標準
    version="0.1.0",
    description="Image utilities (Python port of your Julia ImageUtils)",
    author="Mitsuki ISHIYAMA",
    packages=find_packages(),       # src-layout でないのでこれでOK
    install_requires=[
        "numpy",
        "pillow"
    ],
)
