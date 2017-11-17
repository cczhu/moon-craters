# Input Image Dataset Generator for DeepMoon Convnet Crater Detector
CPS / CITA 
University of Toronto

Scripts and functions for generating the training and validation datasets
for the DeepMoon convolutional neural network-based crater detector.
The detector takes in digital elevation map (DEM) images and pinpoints
the locations of their craters, and their sizes.

To produce training and validation datasets to create the detector,
we cropped small images from the [LRO-Kaguya merged 59 m/pixel DEM](
    https://astrogeology.usgs.gov/search/map/Moon/LRO/LOLA/
Lunar_LRO_LrocKaguya_DEMmerge_60N60S_512ppd). For each input image, we created
a corresponding target image, of the same pixel dimensions, where craters are
represented by binary circular rings.  The rings' centres and radii are
determined from values in the combined LROC 5 - 20 km and Head et al. >=20 km
crater catalogues.

To prepare the DEM for cropping, we used the USGS Astrogeology Cloud Processing
service to convert the DEM from Isis .cub to 16-bit GeoTiff format, and reduce
the resolution to 118 m/pixel.  (Astrocloud had trouble converting directly to
8-bit GeoTiff, mapping values beyond the 8-bit range to either 0 or 255 rather
than rescaling them).  The GDAL library was then used to convert this to an
8-bit png at the same resolution:

```
gdal_translate -of PNG -scale -21138 21138 -co worldfile=no 
    LunarLROLrocKaguya_118mperpix_int16.tif LunarLROLrocKaguya_118mperpix.png
```

The LROC crater catalog was downloaded as shapefiles from the
[LROC site](http://wms.lroc.asu.edu/lroc/rdr_product_select?filter%5Btext%5D=
&filter%5Blat%5D=&filter%5Blon%5D=&filter%5Brad%5D=&filter%5Bwest%5D=
&filter%5Beast%5D=&filter%5Bsouth%5D=&filter%5Bnorth%5D=&filter%5Btopographic
%5D=either&filter%5Bprefix%5D%5B%5D=SHAPEFILE&show_thumbs=0&per_page=
100&commit=Search) and converted to .csv.  The Head et al. catalog was
retrieved from [its Science paper repository](http://science.sciencemag.org/
content/329/5998/1504/tab-figures-data).  The LU78287GT and LU60645GT
catalogus were downloaded from their [USGS Astropedia repository](
    https://astrogeology.usgs.gov/search/map/Moon/Research/Craters/
GoranSalamuniccar_MoonCraters) and converted from .xlsx to .csv.  The Mercury
crater catalogue (currently unused) is from Fassett et al. 2011, and
downloadable from the [Brown Planetary Geosciences Group website](
http://www.planetary.brown.edu/html_pages/mercury_craters.htm).