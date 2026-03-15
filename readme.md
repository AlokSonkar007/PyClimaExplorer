# PyClimaExplorer

Interactive Climate Data Dashboard built with Python and Streamlit.

## What It Does

Loads NetCDF (.nc) climate datasets and lets users explore them through interactive maps and charts.

## Features

**Core:**
- Upload any .nc file (ERA5, CESM, etc.)
- Variable selector and time range filter
- Spatial View -- global heatmap at a selected time slice
- Temporal View -- time-series line plot at a selected location

**Analysis:**
- Z-score anomaly detection with visual markers
- Auto-generated insight cards (trends, peaks, range)
- Anomaly summary table

**Bonus:**
- Comparison Mode -- side-by-side maps of two time periods with difference map
- 3D Globe View -- interactive rotatable globe with color-coded data
- CSV export of filtered data with anomaly flags

## Setup

```bash
pip install streamlit xarray netcdf4 plotly numpy pandas