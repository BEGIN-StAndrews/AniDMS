from setuptools import setup, find_packages

setup(
    name="anidms",
    version="1.1.0",
    description="AniDMS: query monthly DMS data and annotate tracking records with LAEA-IDW",
    author="Meixuan Liu",
    author_email="ml340@st-andrews.ac.uk",
    packages=find_packages(include=["anidms", "anidms.*"]),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "xarray>=0.18.0",
        "scipy>=1.7.0",
        "netCDF4>=1.5.0",
        "requests>=2.25.0",
        "pyproj>=3.0.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
