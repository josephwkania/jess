"""
Install jess.
"""
import glob

from setuptools import find_packages, setup

with open("jess/_version.py", encoding="UTF-8") as f:
    for line in f:
        if "__version__" in line:
            line = line.replace(" ", "").strip()
            version = line.split("__version__=")[1].strip('"')

with open("requirements.txt", encoding="UTF-8") as f:
    required = f.read().splitlines()

setup(
    name="jess",
    version=version,
    packages=find_packages(),
    author="Joseph W Kania",
    scripts=glob.glob("bin/*"),
    install_requires=required,
    extras_require={"cupy": ["cupy>=9.2"]},
)
