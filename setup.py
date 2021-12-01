"""
Install jess.
"""
import glob

from setuptools import find_packages, setup

from jess._version import __version__

with open("requirements.txt", encoding="UTF-8") as f:
    required = f.read().splitlines()

setup(
    name="jess",
    version=__version__,
    packages=find_packages(),
    author="Joseph W Kania",
    scripts=glob.glob("bin/*"),
    install_requires=required,
    extras_require={"cupy": ["cupy>=9.2"]},
)
