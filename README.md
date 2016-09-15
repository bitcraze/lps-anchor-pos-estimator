# crazyflie-anchor-location-estimator [![Build Status](https://api.travis-ci.org/bitcraze/lps-anchor-pos-estimator.svg)](https://travis-ci.org/bitcraze/lps-anchor-pos-estimator)

The aim of this project is to create a python library that uses range samples
from a positioning system as input, for instance the Loco Position System, to 
calculate the position of the anchors.

The goal is to integrate it into the Crazyflie python client, but it can 
probably be useful in other applications as well.

## The plan

The codebase started out as matlab code from Kenneth Batstone at LTH (Lunds 
Tekniska Högskola). The code uses 
[this multipol matlab library](https://github.com/LundUniversityComputerVision/multipol) 
from LTH.

Aman Sharman has done a one to one translation to python based on 
numpy and scipy. The result is the code in this repo. 

The library has not been debugged and verified yet, and might not work at all.

The idea is to refactor the library into something more pythonic when it is 
working. Work in progress...
