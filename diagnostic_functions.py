#!/usr/bin/env python
"""
Suite of test functions for make_input_data.py
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageChops
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import image_slicer as imsl
import glob
import make_input_data as mkin
import timeit

############# Plotting functions #############


def PlotMoonPic(img, craterlist, savefig=False, borderless=True):
    """Matplotlib plot (without using Cartopy) of Moon image
    and crater locations.  Reads in pixel x/y positions rather
    than long/lat for craters.  Useful for comparing Plate Carree
    maps generated by PlotMoonMap.

    Parameters
    ----------
    img : str or ndarray
        Name of file or image file in ndarray form.
    craters : dict
        Dictionary that includes x and y coordinates of crater
        centroids.
    savefig : str or bool
        If true, use as filename.  If False, do not save figure.
    borderless : bool
        Removes whitespace from image
    """

    if type(img) == str:
        img = plt.imread(img)

    if borderless:
        fig = plt.figure(figsize=[img.shape[1]/100., img.shape[0]/100.])
        ax = fig.add_axes([0., 0., 1., 1.])
        ax.set_axis_off()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    ax.imshow(img, origin='upper', cmap="Greys_r")
    if craterlist["x"].size > 0:
        ax.scatter(craterlist["x"], craterlist["y"])
    quiet = ax.axis([0, img.shape[1], img.shape[0], 0])

    if savefig:
        fig.savefig(savefig, dpi = 100, edgecolor = 'w')
        plt.clf()


# Note to self: for tutorials on cartopy, see:
# http://scitools.org.uk/cartopy/docs/v0.13/matplotlib/advanced_plotting.html
# http://scitools.org.uk/cartopy/docs/latest/examples/geostationary.html
# http://scitools.org.uk/iris/docs/v1.9.1/examples/General/projections_and_annotations.html
# https://uoftcoders.github.io/studyGroup/lessons/python/cartography/lesson/

def PlotMoonMap(img="./moonmap_small.png", craters=False, 
                        projection=ccrs.Mollweide(central_longitude=0), savefig=False):
    """Cartopy plot of largest named craters (see code comments for references).
    Reads in long/lat for crater centroids.

    Parameters
    ----------
    img : str or ndarray
        Name of file or image file in ndarray form.
    craters : bool or pandas.DataFrame
        LU78287GT craters.  If false, loads from file.
    mindiam : float
        Minimum crater diameter for inclusion in plot
    projection : cartopy.crs projection
        Map projection
    savefig : str or bool
        If true, use as filename.  If False, do not save figure.
    """

    if type(img) == str:
        img = plt.imread(img)

    # Load craters table
    if not type(craters) == pd.core.frame.DataFrame:
        craters = mkin.ReadSalamuniccarCraterCSV()
    #big_name_craters = craters[(craters["Name"].notnull()) & 
    #                            (craters["Diameter (km)"] > mindiam)]

    fig = plt.figure()

    # Feeding the projection keyword a ccrs method turns ax into a 
    # cartopy.mpl.geoaxes.GeoAxes object, which supports coordinate 
    # system transforms.
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    # As noted in the Iris projections and annotations example, transform 
    # specifies a non-display coordinate system for the data points.
    ax.imshow(img, transform=ccrs.PlateCarree(central_longitude=0), 
                extent=[-180, 180, -90, 90], origin='upper')

    ax.scatter(craters["Long"], craters["Lat"], 
                    transform=ccrs.PlateCarree(central_longitude=0))

    if savefig:
        fig.savefig(savefig, dpi = 200, edgecolor = 'w')


def TrimImageWhitespace(img, outimg):
    """Trims whitespace from image.

    Parameters
    ----------
    img : str
        Name of file.
    outimg : str
        Filename of output.
    """
    im = Image.open(img)
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    im.save(outimg, format="png")


def CentreLongitude(img, outimg):
    """Retiles an image that goes from longitude 0 to 360
    into one that goes from -180 to 180.

    Parameters
    ----------
    img : str
        Name of file.
    outimg : str
        Filename of output.
    """
    im = Image.open(img)
    imo = Image.new(im.mode, im.size, im.getpixel((0,0)))
    # Image midpoint; floor is in case width is odd
    midpt = imo.size[0] // 2
    imo.paste(im, [midpt, 0])
    imo.paste(im, [-imo.size[0] + midpt, 0])
    imo.save(outimg, format="png")


######################################################


############# Test functions #############

def PlotComparison():
    """Script to compare home-brewed and cartopy plots.
    """

    # Note to self: use scipy.ndimage.imread or 
    # Image.open('image.png').convert('LA') if you need 
    # to flatten to greyscale

    img = plt.imread("./moonmap_small.png")
    craters = mkin.ReadSalamuniccarCraterCSV()
    mindiam = 100.

    crater_sub = craters.loc[(craters["Name"].notnull()) & 
                                (craters["Diameter (km)"] > mindiam)]

    cx, cy = coord2pix(crater_sub["Long"].as_matrix(), 
                        crater_sub["Lat"].as_matrix(), 
                        [img.shape[1], img.shape[0]])

    PlotMoonPic(img, {"x": cx, "y": cy}, borderless=True, savefig="./test_moonpic.png")

    PlotMoonMap(img=img, craters=craters, mindiam=mindiam, 
                    projection=ccrs.PlateCarree(central_longitude=0), 
                    savefig="./test_moonmap.png")


def CheckDataSet(imagelist):
    """Overplots csv crater locations onto images made
    by CreateDataSet.
    """

    for imagename in imagelist:
        craters = pd.read_csv(open(imagename.split(".png")[0] + ".csv", 'r'), 
            sep=',', header=0, engine="c", encoding = "ISO-8859-1")
        img = plt.imread(imagename)
        PlotMoonPic(img, craters, savefig=imagename.split(".png")[0] + "_check.png", 
                        borderless=True)


def BigCraters():
    craters = mkin.ReadSalamuniccarCraterCSV(dropfeatures=True)
    big_name_craters = craters[(craters["Name"].notnull()) & \
                                    (craters["Diameter (km)"] > mindiam)].copy()
    mkin.CreatePlateCarreeDataSet("./moonmap_small.png", big_name_craters, 6, outprefix="out/out")

    imagelist = sorted(glob.glob("out/out*.png"))
    CheckDataSet(imagelist)


# Can check against:
# https://tools.wmflabs.org/geohack/geohack.php?pagename=Tycho_%28crater%29&params=43.47_S_16.30_W_globe:moon&title=Tycho+S

def ProminentCrater(cratername="Copernicus r"):
    """Checks if one prominent crater is in the right spot
    in split images.

    Parameters
    ----------
    cratername : str
        Crater name.  Try "Copernicus r" or "Tycho r".
    """

    craters = mkin.ReadSalamuniccarCraterCSV()
    tycho = craters[craters['Name'].str.contains(cratername).fillna(False)].copy()
    CreateDataSet("./moonmap_small.png", tycho, 4, outprefix="out/octr")

    imagelist = sorted(glob.glob("out/octr*.png"))
    CheckDataSet(imagelist)



def ProjectionSpeedTest():
    """Sample speed test; modify to heart's content!"""

    imgname="out/out_01_01.png"
    cdim = [-180, 180, -90, 90]
    csvname="out/out_01_01.csv"
    imgdim=np.array([4292, 2145])
    resolutions=np.array([32, 64, 128, 256, 512, 1024])
    np.random.shuffle(resolutions)
    resolutions = np.r_[np.array([32]), resolutions]

    # Obtain long/lat bounds
    pos = np.array([0, 0])
    size = np.array([429, 214])
    ix = np.array([pos[0], pos[0] + size[0]])
    iy = np.array([pos[1], pos[1] + size[1]])

    # Using origin="upper" means our latitude coordinates are reversed
    llong, llat = mkin.pix2coord(ix, iy, cdim, imgdim, origin="upper")
    llbd = np.r_[llong, llat[::-1]]

    oname = "out/DELETELATER.png"

    imgloadtime = np.mean(timeit.repeat('img = Image.open(imgname).convert("L")', repeat=10,
                        number=1, setup="from __main__ import Image, imgname"))
    raw_image = Image.open(imgname).convert("L")

    restimes = np.zeros(resolutions.size)
    for i, resolution in enumerate(resolutions):
        img = raw_image.resize([resolutions[i],
                int(np.round(resolutions[i] * raw_image.size[1]/raw_image.size[0]))])
        ct = timeit.repeat("MakePCOTransform(img, oname, llbd, csvname)", \
                setup="from __main__ import MakePCOTransform, img, oname, llbd, csvname", \
                number=1, repeat=10)
        restimes[i] = np.mean(ct)

    return imgloadtime, zip(resolutions, restimes)


def MakePCOTransform(img, oname, llbd, csvname):
    """Almost identical to make_input_data.PlateCarree_to_Orthographic,
    except designed to test relevant loading times when creating large
    datasets.
    """

    origin = "upper"
    rgcoeff = 1.2
    ctr_sub = True
    iglobe = ccrs.Globe(semimajor_axis=1737400, 
                    semiminor_axis=1737400,
                    ellipse=None)

    craters = pd.read_csv(csvname, 
        sep=',', header=0, engine="c", encoding = "ISO-8859-1")

    # Set up Geodetic (long/lat), Plate Carree (usually long/lat, but
    # not when globe != WGS84) and Orthographic projections
    geoproj = ccrs.Geodetic(globe=iglobe)
    iproj = ccrs.PlateCarree(globe=iglobe)
    oproj = ccrs.Orthographic(central_longitude=np.mean(llbd[:2]), 
                            central_latitude=np.mean(llbd[2:]), 
                            globe=iglobe)

    # Create and transform coordinates of image corners and
    # edge midpoints.  Due to Plate Carree and Orthographic's symmetries,
    # max/min x/y values of these 9 points represent extrema
    # of the transformed image.
    xll = np.array([llbd[0], np.mean(llbd[:2]), llbd[1]])
    yll = np.array([llbd[2], np.mean(llbd[2:]), llbd[3]])
    xll, yll = np.meshgrid(xll, yll)
    xll = xll.ravel(); yll = yll.ravel()

    # [:,:2] becaus we don't need elevation data
    res = iproj.transform_points(x=xll, y=yll,
                                src_crs=geoproj)[:,:2]
    iextent = [min(res[:,0]), max(res[:,0]), 
                min(res[:,1]), max(res[:,1])]

    res = oproj.transform_points(x=xll, y=yll,
                                src_crs=geoproj)[:,:2]
    oextent = [min(res[:,0]), max(res[:,0]), 
                min(res[:,1]), max(res[:,1])] 

    imgo, imgwshp, offset = mkin.WarpImagePad(img, iproj, iextent, 
                    oproj, oextent, origin=origin, rgcoeff=rgcoeff, 
                    fillbg="white")

    # Convert crater x, y position
    if ctr_sub:
        llbd_in = None
    else:
        llbd_in = llbd
    ctr_xy = mkin.WarpCraterLoc(craters, geoproj, oproj, 
                oextent, img.size, llbd=llbd_in)
    # Shift crater x, y positions by offset
    # (origin doesn't matter for y-shift, since
    # padding is symmetric)
    ctr_xy.loc[:, "x"] += offset[0]
    ctr_xy.loc[:, "y"] += offset[1]

    imgo.save(oname)
    ctr_xy.to_csv(oname.split(".png")[0] + ".csv", index=False)


#if __name__ == "__main__":
    # execute only if run as a script
    #BigCraters()
