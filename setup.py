version = "0.0.1"

import glob
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="jess",
    version=version,
    author="Joseph W Kania",
    scripts=glob.glob("bin/*"),
    install_requirements=required,
)
