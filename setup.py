import os
from setuptools import setup

README = open(os.path.join(os.path.dirname(__file__), "README.rst")).read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))
# get version information
exec(open("openrem/remapp/version.py").read())

with open("requirements.txt") as f:
    REQUIRED = f.read().splitlines()


setup(
    name="OpenREM",
    version=__version__,
    packages=["openrem"],
    include_package_data=True,
    install_requires=REQUIRED,
    scripts=[
        "openrem/scripts/openrem_rdsr.py",
        "openrem/scripts/openrem_mg.py",
        "openrem/scripts/openrem_dx.py",
        "openrem/scripts/openrem_ctphilips.py",
        "openrem/scripts/openrem_cttoshiba.py",
        "openrem/scripts/openrem_ptsizecsv.py",
        "openrem/scripts/openrem_qr.py",
        "openrem/scripts/openrem_nm.py",
    ],
    license="GPLv3 with additional permissions",
    # description="Open source patient radiation dose monitoring software",
    description='Developer beta only',
    long_description=README,
    url="https://openrem.org/",
    author="Ed McDonagh",
    author_email="ed@openrem.org",
    long_description_content_type="text/x-rst",
    classifiers=[
        "Environment :: Web Environment",
        "Framework :: Django",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
