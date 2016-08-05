#!/usr/bin/env python3
from setuptools import find_packages
from setuptools import setup

setup(
    name='lpsposest',
    version='0.1.0',
    packages=find_packages(exclude=['examples', 'tests']),

    description='Loco Positioning anchor position estimator',
    url='https://github.com/bitcraze/lps-anchor-pos-estimator',

    author='Bitcraze and contributors',
    license='',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: System :: Hardware :: Hardware Drivers',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3'
    ],

    keywords='loco positioning estimator',

    install_requires='',
)
