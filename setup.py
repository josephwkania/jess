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
    description="Just in Time Elimination of Spurious Signals - jess",
    url="https://github.com/josephwkania/jess",
    author="Joseph W Kania",
    packages=find_packages(),
    scripts=glob.glob("bin/*"),
    python_requires=">=3.6, <4",
    install_requires=required,
    extras_require={
        "tests": ["pytest", "pytest-cov"],
        "cupy": ["cupy>=9.2"],
        "docs": ["sphinx", "myst-parser"],
    },
)
