from setuptools import setup, find_packages

# Package metadata
NAME = 'entss'
VERSION = '0.1.0'
DESCRIPTION = 'Entailment Classification and Semantic Scaling'
URL = 'https://github.com/MLBurnham/entss'
AUTHOR = 'Michael Burnham'
AUTHOR_EMAIL = 'mlb6496@psu.edu'
LICENSE = 'GPL-3.0'

# Read the long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

# Define package dependencies
INSTALL_REQUIRES = [
    "pandas >= 1.2.3",
    "numpy >= 1.21.5",
    "pysbd >= 0.3.4",
    "regex >= 2022.3.15",
    "cmdstanpy >= 1.1.0",
    "transformers >= 4.0.0",
    "torch >= 2.0.0"
]

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3'
    ],
    include_package_data=True,
    package_data={'':['data/*.csv', 'models/*.stan']}
)