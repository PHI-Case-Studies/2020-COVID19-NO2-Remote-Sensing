#!/bin/bash
mkdir -p data/gadm36
wget https://biogeo.ucdavis.edu/data/gadm3.6/gadm36_levels_shp.zip -O data/gadm36/gadm36_levels_shp.zip
unzip -o data/gadm36/gadm36_levels_shp.zip -d data/gadm36
rm data/gadm36/gadm36_levels_shp.zip
ls -la data/gadm36/
