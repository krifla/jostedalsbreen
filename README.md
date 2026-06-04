# Brief description
Code to analyse WRF model output saved as 2D fields or time series.

# Structure
The code uses main.py to call:
1. loadData.py - for loading observations and model output
2. weighted4pts.py - for calculating inverse distance weighted averages of the four closest grid cell for comparison to observations
3. plotData.py - for plotting some of the data

# Related publications
Figures produced and numbers estimated with this code are used in Haualand et al. (accepted in WCD) - DOI TO BE ADDED!

Model output data is also published at https://doi.org/10.5281/zenodo.20529869.

