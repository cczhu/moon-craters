#!/usr/bin/env python
"""Input Image Dataset Generator Functions

Functions for generating input and target image datasets from Lunar digital
elevation maps and crater catalogues.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from PIL import Image
import cartopy.crs as ccrs
import cartopy.img_transform as cimg
import image_slicer as imsl
import glob
import collections
import pickle
import re
import cv2
from scipy.spatial import cKDTree as kd

########## Read Cratering CSVs ##########

def ReadLROCCraterCSV(filename="./LROCCraters.csv", sortlat=True):
    """Reads LROC 5 - 20 km crater catalogue CSV obtained by Alan Jackson.

    Parameters
    ----------
    filename : str, optional
        Filepath and name of LROC csv file.  Default assumes it exists in the
        current working directory.
    sortlat : bool, optional
        If `True` (default), order catalogue by latitude.

    Returns
    -------
    craters : pandas.DataFrame
        Craters data frame.
    """
    craters = pd.read_csv(filename, header=0, usecols=list(range(2, 6)))
    if sortlat:
        craters.sort_values(by='Lat', inplace=True)
        craters.reset_index(inplace=True, drop=True)

    return craters


def ReadHeadCraterCSV(filename="./HeadCraters.csv", sortlat=True):
    """Reads Head et al. 2010 (`http://adsabs.harvard.edu/
    abs/2010Sci...329.1504H`) >= 20 km diameter crater catalogue.

    Parameters
    ----------
    filename : str, optional
        Filepath and name of Head et al. csv file.  Default assumes it exists
        in the current working directory.
    sortlat : bool, optional
        If `True` (default), order catalogue by latitude.

    Returns
    -------
    craters : pandas.DataFrame
        Craters data frame.
    """
    craters = pd.read_csv(filename, header=0,
                          names=['Long', 'Lat', 'Diameter (km)'])
    if sortlat:
        craters.sort_values(by='Lat', inplace=True)
        craters.reset_index(inplace=True, drop=True)

    return craters


def ReadLROCHeadCombinedCraterCSV(filelroc="./LROCCraters.csv",
                                  filehead="./LolaLargeCraters.csv",
                                  sortlat=True):
    """Combines LROC 5 - 20 km crater dataset with Head >= 20 km dataset.

    Parameters
    ----------
    filelroc : str, optional
        LROC crater file location
    filehead : str, optional
        Head et al. crater file location
    sortlat : bool, optional
        If `True` (default), order catalogue by latitude.

    Returns
    -------
    craters : pandas.DataFrame
        Craters data frame.
    """
    ctrs_head = ReadHeadCraterCSV(filename=filehead, sortlat=False)
    ctrs_head = ctrs_head[ctrs_head["Diameter (km)"] > 20]
    ctrs_lroc = ReadLROCCraterCSV(filename=filelroc, sortlat=False)
    ctrs_lroc.drop(["tag"], axis=1, inplace=True)
    craters = pd.concat([ctrs_lroc, ctrs_head], axis=0, ignore_index=True,
                        copy=True)
    craters.sort_values(by='Lat', inplace=True)
    craters.reset_index(inplace=True, drop=True)

    return craters


def ReadSalamuniccarCraterCSV(filename="./LU78287GT.csv", dropfeatures=False,
                              sortlat=True):
    """Reads LU78287GT (Salamuniccar et al. 2014) crater file CSV, converted
    from the xlsx file found at `https://astrogeology.usgs.gov/search/map/
    Moon/Research/Craters/GoranSalamuniccar_MoonCraters`.  That file also
    contains LU60645GT, which is also compatible with this function.

    Parameters
    ----------
    filename : str
        csv file of craters
    dropfeatures : bool, optional
        If true, drop satellite craters (those listed with "A", "B", "C"...),
        leaving only the whole crater (listed as "r" or without a second
        letter). Only useful if you want to (crudely) remove secondary impacts.
    sortlat : bool, optional
        If `True`, order catalogue by latitude.

    Returns
    -------
    craters : pandas.DataFrame
        Craters data frame.
    """
    # Read in crater names
    craters_names = ["ID", "Long", "Lat", "Radius (deg)",
                     "Diameter (km)", "D_range", "p", "Name"]
    craters_types = [str, float, float, float, float, float, int, str]
    craters = pd.read_csv(
        open(filename, 'r'), sep=',', usecols=list(range(8)), header=0,
        engine="c", encoding="ISO-8859-1", names=craters_names,
        dtype=dict(zip(craters_names, craters_types)))

    # Truncate cyrillic characters
    craters["Name"] = craters["Name"].str.split(":").str.get(0)

    if dropfeatures:
        DropSatelliteCraters(craters)

    if sortlat:
        craters.sort_values(by='Lat', inplace=True)
        craters.reset_index(inplace=True, drop=True)

    return craters


def DropSatelliteCraters(craters):
    """Drops named crater sub-features (listed with "A", "B", "C"...), leaving
    only the whole crater (listed as "r" or with no second letter).

    Parameters
    ----------
    craters : pandas.DataFrame
        Craters data frame to be cleaned of features.
    """

    # String matching thingy
    def match_end(s):
        if re.match(r" [A-Z]", s[-2:]):
            return True
        return False

    # Find all crater names that ends with A - Z
    basenames = (
        craters.loc[craters["Name"].notnull(), "Name"].apply(match_end))
    drop_index = basenames[basenames].index
    craters.drop(drop_index, inplace=True)


def ReadLROCLUCombinedCraterCSV(filealan="./LROCCraters.csv",
                                filelu="./LU78287GT.csv",
                                dropfeatures=False):
    """Combines LROC 5 - 20 km crater dataset with Goran Salamuniccar craters
    that are > 20 km.

    Parameters
    ----------
    filealan : str
        LROC crater file location
    filelu : str
        Salamuniccar crater file location
    dropfeatures : bool
        If true, drop satellite craters (those listed with
        "A", "B", "C"...), leaving only the whole crater
        (listed as "r").  Only useful if you want to (crudely)
        remove secondary impacts.

    Returns
    -------
    craters : pandas.DataFrame
        Craters data frame.
    """

    # Read in LU crater names
    craters_names = ["Long", "Lat", "Radius (deg)",
                     "Diameter (km)", "D_range", "p", "Name"]
    craters_types = [float, float, float, float, float, int, str]
    craters = pd.read_csv(
        open(filelu, 'r'), sep=',', usecols=list(range(1, 8)), header=0,
        engine="c", encoding="ISO-8859-1", names=craters_names,
        dtype=dict(zip(craters_names, craters_types)))

    # Truncate cyrillic characters
    craters["Name"] = craters["Name"].str.split(":").str.get(0)

    if dropfeatures:
        DropSatelliteCraters(craters)

    craters.drop(["Radius (deg)", "D_range", "p", "Name"],
                 axis=1, inplace=True)
    craters = craters[craters["Diameter (km)"] > 20]

    craters_alan = pd.read_csv(filealan, header=0, usecols=list(range(2, 5)))

    craters = pd.concat([craters, craters_alan], axis=0, ignore_index=True,
                        copy=True)

    craters.sort_values(by='Lat', inplace=True)
    craters.reset_index(inplace=True, drop=True)

    return craters


def ReadMercuryCraterCSV(filename="./MercLargeCraters.csv", sortlat=True):
    """Reads crater file CSV from Fassett et al.
    (http://www.planetary.brown.edu/html_pages/mercury_craters.htm)

    Parameters
    ----------
    filename : str
        csv file of craters

    Returns
    -------
    craters : pandas.DataFrame
        Craters data frame.
    """

    craters_names = ["Long", "Lat", "Diameter (km)"]
    craters_types = [float, float, float]
    craters = pd.read_csv(
        open(filename, 'r'), sep=',', header=0, names=craters_names,
        dtype=dict(zip(craters_names, craters_types)))

    if sortlat:
        craters.sort_values(by='Lat', inplace=True)
        craters.reset_index(inplace=True, drop=True)

    return craters

########## Coordinates to pixels projections ##########

def coord2pix(cx, cy, cdim, imgdim, origin="upper"):
    """Converts coordinate x/y to image pixel locations.

    Parameters
    ----------
    cx : float or ndarray
        Coordinate x
    cy : float or ndarray
        Coordinate y
    cdim : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of image
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels
    origin : "upper" or "lower"
        Based on imshow convention for displaying image y-axis.
        "upper" means that [0,0] is upper-left corner of image;
        "lower" means it is bottom-left.

    Returns
    -------
    x : float or ndarray
        pixel x positions
    y : float or ndarray
        pixel y positions
    """

    x = imgdim[0] * (cx-cdim[0]) / (cdim[1]-cdim[0])

    if origin == "lower":
        y = imgdim[1] * (cy-cdim[2])/(cdim[3]-cdim[2])
    else:
        y = imgdim[1] * (cdim[3]-cy)/(cdim[3]-cdim[2])

    return x, y


def pix2coord(x, y, cdim, imgdim, origin="upper"):
    """Converts image pixel locations to Plate Carree lat/long.
    Assumes central meridian is at 0 (so long in [-180, 180) ).

    Parameters
    ----------
    x : float or ndarray
        pixel x positions
    y : float or ndarray
        pixel y positions
    cdim : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of image
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels
    origin : "upper" or "lower"
        Based on imshow convention for displaying image y-axis.
        "upper" means that [0,0] is upper-left corner of image;
        "lower" means it is bottom-left.

    Returns
    -------
    cx : float or ndarray
        Coordinate x
    cy : float or ndarray
        Coordinate y
    """

    cx = (x/imgdim[0]) * (cdim[1]-cdim[0]) + cdim[0]

    if origin == "lower":
        cy = (y/imgdim[1]) * (cdim[3]-cdim[2]) + cdim[2]
    else:
        cy = cdim[3] - (y/imgdim[1]) * (cdim[3]-cdim[2])

    return cx, cy

########## Metres to Pixels ##########

def km2pix(imgheight, latextent, dc=1., a=1737.4):
    """Returns conversion from km to pixels.

    Parameters
    ----------
    imgheight : float
        Height of image in pixels
    latextent : float
        Latitude extent of image in degrees
    dc : float from 0 to 1
        Scaling factor for distortions
    a : float
        World radius in km.  Default is Moon (1737.4 km)

    Returns
    -------
    km2pix : float
        Conversion factor pix/km
    """
    return (180./np.pi)*imgheight*dc/latextent/a

########## Warp Images and CSVs ##########

def regrid_shape_aspect(regrid_shape, target_extent):
    """
    Helper function copied from cartopy.img_transform for 
    setting regridding shape which is used in several
    plotting methods.
    """
    if not isinstance(regrid_shape, collections.Sequence):
        target_size = int(regrid_shape)
        x_range, y_range = np.diff(target_extent)[::2]
        desired_aspect = x_range / y_range
        if x_range >= y_range:
            regrid_shape = (target_size * desired_aspect, target_size)
        else:
            regrid_shape = (target_size, target_size / desired_aspect)
    return regrid_shape


def WarpImage(img, iproj, iextent, oproj, oextent,
                origin="upper", rgcoeff=1.2):
    """
    Warps images with cartopy.img_transform.warp_array,
    then plots them with imshow.  Based on
    cartopy.mpl.geoaxes.imshow.  Parameter descriptions
    are identical to those in WarpImagePad.
    """

    if iproj == oproj:
        raise Warning("WARNING: input and output transforms are identical!"
                       "Returing input!")
        return img

    else:

        if origin == 'upper':
            # Regridding operation implicitly assumes origin of 
            # image is 'lower', so adjust for that here.
            img = img[::-1]

        # The 1.2 is padding when we rescale the image with imshow
        regrid_shape = rgcoeff*min(img.shape)
        regrid_shape = regrid_shape_aspect(regrid_shape,
                                         oextent)

        # cimg.warp_array uses cimg.mesh_projection, which
        # cannot handle any zeros being used in iextent.  Create
        # iextent_nz to fix
        iextent_nz = np.array(iextent)
        iextent_nz[iextent_nz == 0] = 1e-8
        iextent_nz = list(iextent_nz)

        imgout, extent = cimg.warp_array(img,
                             source_proj=iproj,
                             source_extent=iextent_nz,
                             target_proj=oproj,
                             target_res=regrid_shape,
                             target_extent=oextent,
                             mask_extrapolated=True)

        if origin == 'upper':
            # Transform back
            imgout = imgout[::-1]

        return imgout


# https://stackoverflow.com/questions/2563822/how-do-you-composite-an-image-onto-another-image-with-pil-in-python

def WarpImagePad(img, iproj, iextent, oproj, oextent, 
                origin="upper", rgcoeff=1.2, fillbg="white"):
    """
    Wrapper for WarpImage that adds padding to warped
    image to make it the same size as the original.

    Parameters
    ----------
    img : numpy.ndarray
        Image as a 2D array
    iproj : cartopy.crs.Projection instance
        Input coordinate system
    iextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) 
        of input
    oproj : cartopy.crs.Projection instance
        Output coordinate system
    oextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) 
        of output
    origin : "lower" or "upper"
        Based on imshow convention for displaying image y-axis.
        "upper" means that [0,0] is upper-left corner of image;
        "lower" means it is bottom-left.
    rgcoeff : float
        Fractional size increase of transformed image height;
        generically set to 1.2 to prevent loss of fidelity
        during transform (though warping can be so extreme
        that this might be meaningless)      

    Returns
    -------
    imgo : PIL.Image.Image
        Warped image with padding
    imgw.size : tuple
        Width, height of picture without padding
    offset : tuple
        Pixel width of (left, top)-side padding
    """

    if type(img) == Image.Image:
        img = np.asanyarray(img)

    # Set background colour
    if fillbg == "white":
        bgval = 255
    else:
        bgval = 0

    # Warp image
    imgw = WarpImage(img, iproj, iextent, oproj, oextent, 
                origin=origin, rgcoeff=rgcoeff)

    # Remove mask, turn image into Image.Image
    imgw = np.ma.filled(imgw, fill_value=bgval)
    imgw = Image.fromarray(imgw, mode="L")

    # Resize to height of original, maintaining
    # aspect ratio.  Note img.shape = height, width, and
    # imgw.size and imgo.size = width, height
    imgw_loh = imgw.size[0] / imgw.size[1]

    # If imgw is stretched horizontally compared to img
    if imgw_loh > img.shape[1]/img.shape[0]:
        imgw = imgw.resize([img.shape[0], 
                    int(np.round(img.shape[0] / imgw_loh))])
    # If imgw is stretched vertically
    else:
        imgw = imgw.resize([int(np.round(imgw_loh*img.shape[0])), 
                    img.shape[0]])

    # Make background image and paste two together
    imgo = Image.new('L', (img.shape[1], img.shape[0]), 
                        (bgval))
    offset = ((imgo.size[0] - imgw.size[0]) // 2, 
                (imgo.size[1] - imgw.size[1]) // 2)
    imgo.paste(imgw, offset)

    return imgo, imgw.size, offset


def WarpCraterLoc(craters, geoproj, oproj, 
                oextent, imgdim, llbd=None,
                origin="upper"):
    """
    Wrapper for WarpImage that adds padding to warped
    image to make it the same size as the original.

    Parameters
    ----------
    craters : pandas.DataFrame
        Crater info
    geoproj : cartopy.crs.Geodetic instance
        Input lat/long coordinate system
    oproj : cartopy.crs.Projection instance
        Output coordinate system
    oextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) 
        of output
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels
    llbd : list-like
        Long/lat limits (long_min, long_max, 
        lat_min, lat_max) of image
    origin : "lower" or "upper"
        Based on imshow convention for displaying image y-axis.
        "upper" means that [0,0] is upper-left corner of image;
        "lower" means it is bottom-left.

    Returns
    -------
    ctr_wrp : pandas.DataFrame
        DataFrame that includes pixel x, y positions
    """

    # Get subset of craters within llbd limits
    if llbd is None:
        ctr_wrp = craters
    else:
        ctr_xlim = (craters["Long"] >= llbd[0]) & \
                    (craters["Long"] <= llbd[1])
        ctr_ylim = (craters["Lat"] >= llbd[2]) & \
                    (craters["Lat"] <= llbd[3])
        ctr_wrp = craters.loc[ctr_xlim & \
                                ctr_ylim, :].copy()

    # Get output projection coords.
    # [:,:2] becaus we don't need elevation data
    # If statement is in case ctr_wrp has nothing in it
    if ctr_wrp.shape[0]:
        ilong = ctr_wrp["Long"].as_matrix()
        ilat = ctr_wrp["Lat"].as_matrix()
        res = oproj.transform_points(x=ilong, y=ilat,
                                    src_crs=geoproj)[:,:2]

        # Get output
        ctr_wrp["x"], ctr_wrp["y"] = coord2pix(res[:,0], 
                        res[:,1], oextent, imgdim, 
                        origin=origin)
    else:
        ctr_wrp["x"] = []
        ctr_wrp["y"] = []

    return ctr_wrp



############# Warp Plate Carree to Orthographic ###############


def PlateCarree_to_Orthographic(img, oname, llbd, craters, 
                                iglobe=None, ctr_sub=False,
                                arad=1737.4, origin="upper", 
                                rgcoeff=1.2, slivercut=0.,
                                dontsave=False):
    """Transform Plate Carree image and associated csv file 
    into Orthographic

    Parameters
    ----------
    img : PIL.Image.image or str
        File or filename
    oname : str
        Output filename
    llbd : list-like
        Long/lat limits (long_min, long_max, 
        lat_min, lat_max) of image
    craters : pandas.DataFrame
        Craters dataframe
    iglobe : cartopy.crs.Geodetic instance
        Globe for images.  If False, defaults to spherical Moon.
    ctr_sub : bool
        If True, assumes craters dataframe includes only craters
        within image.  If False, llbd used to cut craters
        from outside image out of (copy of) dataframe.
    arad : float
        World radius in km.  Default is Moon (1737.4 km)
    origin : "lower" or "upper"
        Based on imshow convention for displaying image y-axis.
        "upper" means that [0,0] is upper-left corner of image;
        "lower" means it is bottom-left.
    rgcoeff : float
        Fractional size increase of transformed image height;
        generically set to 1.2 to prevent loss of fidelity
        during transform (though warping can be so extreme
        that this might be meaningless)
    slivercut : float from 0 to 1
        If transformed image aspect ratio is to narrow (and would
        lead to a lot of padding, return null images)
    dontsave : bool
        To save or not to save, that is the queseiton.

    Returns (only if dontsave = True)
    ---------------------------------
    imgo : PIL.Image.image
        Transformed, padded image in PIL.Image format
    ctr_xy : pandas.DataFrame
        Craters with transformed x, y pixel positions and
        pixel radii
    """

    # If user doesn't provide moon globe properties
    if not iglobe:
        iglobe = ccrs.Globe(semimajor_axis=arad*1000., 
                        semiminor_axis=arad*1000.,
                        ellipse=None)

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

    # Sanity check for narrow images; done before
    # the most expensive part of function
    oaspect = (oextent[1] - oextent[0]) / (oextent[3] - oextent[2])
    if oaspect < slivercut:
        if dontsave:
            return [None, None]
        return

    if type(img) != Image.Image:
        img = Image.open(img).convert("L")

    imgo, imgwshp, offset = WarpImagePad(img, iproj, iextent, 
                    oproj, oextent, origin=origin, rgcoeff=rgcoeff, 
                    fillbg="black")

    # Convert crater x, y position
    if ctr_sub:
        llbd_in = None
    else:
        llbd_in = llbd
    ctr_xy = WarpCraterLoc(craters, geoproj, oproj, 
                oextent, imgwshp, llbd=llbd_in,
                origin=origin)
    # Shift crater x, y positions by offset
    # (origin doesn't matter for y-shift, since
    # padding is symmetric)
    ctr_xy.loc[:, "x"] += offset[0]
    ctr_xy.loc[:, "y"] += offset[1]

    # Pixel scale for orthographic determined (for images small enough
    # that tan(x) approximately equals x + 1/3x^3 + ... remember you
    # can check size of next expansion term!) by l = R_moon*theta,
    # where theta is the latitude extent of the centre of the image.
    # Because projection transform doesn't guarantee central axis
    # will keep its pixel resolution, we need to calculate the
    # conversion coefficient C = (res[7,1] - res[1,1])/(oextent[3] - oextent[2])
    # C0*pix height/C = theta (theta = latitude extent; C0
    # is the theta per pixel conversion for the Plate Carree image).
    # Thus l_ctr = R_moon*C0*pix_ctr/C.
    Cd = (res[7,1] - res[1,1])/(oextent[3] - oextent[2])
    if Cd < 0.7:
        raise AssertionError("Cd cannot be {0:.2f}!".format(Cd))
    pxperkm = km2pix(imgo.size[1], llbd[3] - llbd[2], \
                        dc=Cd, a=arad)
    ctr_xy["Diameter (pix)"] = ctr_xy["Diameter (km)"] * pxperkm

    if dontsave:
        return [imgo, ctr_xy]

    imgo.save(oname)
    ctr_xy.to_csv(oname.split(".png")[0] + ".csv", index=False)


############# Create Tiled Orthographic Dataset #############


def AddPlateCarree_XY(craters, imgdim, cdim=[-180, 180, -90, 90], 
                        origin="upper"):
    """Adds x and y pixel locations to craters dataframe.

    Parameters
    ----------
    craters : pandas.DataFrame
        Crater info
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels
    origin : "upper" or "lower"
        Based on imshow convention for displaying image y-axis.
        "upper" means that [0,0] is upper-left corner of image;
        "lower" means it is bottom-left.
    """
    x, y = coord2pix(craters["Long"].as_matrix(), craters["Lat"].as_matrix(), 
                        cdim, imgdim, origin=origin)
    craters["x"] = x
    craters["y"] = y


def CreateLunarPlateCarreeDataSet(img, craters, splitnum, outprefix="out",
                            savecoords=False):
    """Creates set of images and accompanying csvs of crater data.

    Parameters
    ----------
    img : str
        Name of file.
    craters : pandas.DataFrame
        Crater dataframe  
    splitnum : int
        Number of subfiles to split image into.
    outprefix : str
        Output files' prefix.
    savecoords : bool
        If True, saves tile coordinates
    """

    tiles = imsl.slice(img, splitnum, save=False)
    imgshape = list(imsl.get_combined_size(tiles))

    # Origin is upper for image_slicer, and shapes
    # are x-axis ("columns") first, rather than
    # rows as in plt.imread
    AddPlateCarree_XY(craters, imgshape)

    for tile in tiles:
        # Get x, y limits of image
        ctr_xlim = (craters["x"] > tile.coords[0]) & \
                    (craters["x"] < tile.coords[0] + tile.image.size[0])
        ctr_ylim = (craters["y"] > tile.coords[1]) & \
                    (craters["y"] < tile.coords[1] + tile.image.size[1])

        # Get subset of craters within these limits
        curr_craters = craters.loc[ctr_xlim & ctr_ylim, :].copy()
        curr_craters.loc[:,"x"] -= tile.coords[0]
        curr_craters.loc[:,"y"] -= tile.coords[1]

        # Obtain output image name
        outname = tile.generate_filename(prefix=outprefix,
                          format='png', path=True)

        tile.save(outname, format="png")
        curr_craters.to_csv(outname.split(".png")[0] + ".csv", index=False)

    if savecoords:
        tileval = []
        tilename = []

        for t in tiles:
            tileval.append({"pos": t.position, "coord": t.coords,
                            "num": t.number, "size": t.image.size})
            tfilename = t.generate_filename(prefix=outprefix)
            tilename.append(tfilename.split("/")[-1])

        tdict = dict(zip(tilename, tileval))
        pickle.dump(tdict, open(tfilename.split(outprefix)[0] + \
                    outprefix + "_tiles.p", 'wb'))


def CreateLunarOrthographicDataSet(outprefix):
    """Creates set of Moon images and accompanying csvs of crater data from
    a set of Plate Carree data.

    Parameters
    ----------
    outprefix : str
        Plate Carree output files' filepath and image prefix.
    """

    imagelist = sorted(glob.glob(outprefix + "*.png"))
    tdict = pickle.load(open(glob.glob(outprefix + "_tiles.p")[0], 'rb'))

    #lastimg = Image.open(imagelist[-1]).convert("L")
    lastimg = tdict[imagelist[-1].split("/")[-1]]
    imgdim = tuple( np.array(lastimg["coord"]) + \
                    np.array(lastimg["size"]) )

    cdim = [-180, 180, -90, 90]
    prefix_head = outprefix.split("/")[-1]

    for item in imagelist:

        # Obtain long/lat bounds
        pos = np.array(tdict[item.split("/")[-1]]["coord"])
        size = np.array(tdict[item.split("/")[-1]]["size"])
        ix = np.array([pos[0], pos[0] + size[0]])
        iy = np.array([pos[1], pos[1] + size[1]])

        # Using origin="upper" means our latitude coordinates are reversed
        llong, llat = pix2coord(ix, iy, cdim, imgdim, origin="upper")
        llbd = np.r_[llong, llat[::-1]]

        craters = pd.read_csv(open(item.split(".png")[0] + ".csv", 'r'), 
            sep=',', header=0, engine="c", encoding = "ISO-8859-1")

        oname = item.split( "/" + prefix_head )[0] + "/" + \
                "ortho" + item.split( "/" + prefix_head )[1]

        PlateCarree_to_Orthographic(item, oname, llbd, craters, 
                                    iglobe=None, ctr_sub=True,
                                    arad=1737.4, origin="upper",
                                    rgcoeff=1.2)


################### Create Random Dataset ###########################


#def RandRot(img, craters, expand=False, origin="upper"):
#    """Rotates and horizontally/vertically flips image at random.
#    DOES NOT SEED ITSELF!"""

#    # Values to shift craters after rotation
#    if origin == "upper":
#        rot_shift = [(0, img.size[0]), (img.size[0], img.size[1]), 
#                            (img.size[1],0)]
#    else:
#        rot_shift = [(img.size[1],0), (img.size[0], img.size[1]), 
#                            (0, img.size[0])]

#    rtoken = np.random.randint(0,4)
#    if rtoken > 0:
#        img = img.rotate(90*rtoken, expand=expand)

#        # Rotate crater coordinates
#        if origin == "upper":
#            rt = -rtoken
#        else:
#            rt = rtoken
#        costheta = np.cos(90.*rt/180.*np.pi)
#        sintheta = np.sin(90.*rt/180.*np.pi)
#        xrot = craters["x"]*costheta - craters["y"]*sintheta
#        yrot = craters["x"]*sintheta + craters["y"]*costheta
#        craters["x"] = xrot + rot_transform[rtoken - 1][0]
#        craters["y"] = yrot + rot_transform[rtoken - 1][1]

#    # 50% chance of flipping or mirroring
#    if np.random.randint(0,2):
#        img = ImageOps.mirror(img)
#        craters["x"] = img.size[0] - craters["x"]

#    if np.random.randint(0,2):
#        img = ImageOps.flip(img)
#        craters["y"] = img.size[1] - craters["y"]

#    return [img, craters]


def ResampleCraters(craters, llbd, imgheight, arad=1737.4, minpix=0):
    """Crops crater file, and removes craters smaller than 
    some user defined minimum value.

    Parameters
    ----------
    craters : pandas.DataFrame
        Crater dataframe
    llbd : list-like
        Long/lat limits (long_min, long_max, 
        lat_min, lat_max) of image
    imgheight : int
        Pixel height of image
    arad : float
        World radius in km.  Defaults to Moon radius (1737.4 km)
    minpix : int
        Minimium crater pixel size to be included
        in output

    Returns
    -------
    ctr_sub : pandas.DataFrame
        Cropped and filtered dataframe
    """

    # Get subset of craters within llbd limits
    ctr_xlim = (craters["Long"] >= llbd[0]) & \
                (craters["Long"] <= llbd[1])
    ctr_ylim = (craters["Lat"] >= llbd[2]) & \
                (craters["Lat"] <= llbd[3])
    ctr_sub = craters.loc[ctr_xlim & \
                            ctr_ylim, :].copy()

    if minpix > 0:
        # Obtain pixel per km conversion factor.  Use
        # latitude because Plate carree doesn't distort
        # along this axis
        pxperkm = km2pix(imgheight, llbd[3] - llbd[2], \
                            dc=1., a=arad)
        minkm = minpix / pxperkm
        
        # Remove craters smaller than pixel limit
        ctr_sub = ctr_sub[ctr_sub["Diameter (km)"] >= minkm]
        ctr_sub.reset_index(inplace=True, drop=True)
    
    return ctr_sub


#def GenDatasetNP(img, ilen=200, randrot=True, amt=100):
#    """Generates random dataset.

#    Parameters
#    ----------
#    """
#    xm, ym = img.shape[1] - ilen, img.shape[0] - ilen

#    for i in range(amt):
#        xc = np.random.randint(0, xm)
#        yc = np.random.randint(0, ym)

#        im = img[yc:yc + ilen, xc:xc + ilen]

#        if min(im.shape) == 0:
#            print(xc, yc, ilen)

#        yield im.copy()


def GenDataset(img, craters, outhead, ilen_range=np.array([300., 4000.]), 
                olen=300, cdim=[-180, 180, -90, 90], arad=1737.4,
                minpix=0, amt=100, zeropad=4, slivercut=0.1, 
                outp=False, istart = 0, seed=None):
    """Generates random dataset from plate image.

    The function randomly samples small images from a Plate Carree
    projection global digital elevation map, and converts the images to
    Orthographic projection.  Pixel coordinates and radii of craters in each
    image are derived from a catalogue.  Images and Pandas tables of
    corresponding craters are then saved to disk in hdf5 format.

    Parameters
    ----------
    img : PIL.Image.Image
        Source image
    craters : pandas.DataFrame
        Crater list csv
    outhead : str
        Filepath and file prefix to save output
        images under.
    ilen_range : list-like
        Lower and upper bounds of image width, in pixels,
        to crop from source.  To always crop the same sized
        image, set lower bound to same value as upper.
    olen : int
        Output image width, in pixels.  Cropped images will be
        downsampled to this size.
    cdim : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of image
    arad : float
        World radius in km.  Defaults to Moon radius (1737.4 km)
    minpix : int
        Minimum crater diameter in pixels to be included in
        crater list.  By default, not useful, since
        our source image is ~100-200 m/px, we downsample by
        at most a factor of 10 and our crater dataset starts at
        d = 5000 m.  However, if you use a different image, or
        setting max(ilen_range) > 10000 px, might be necessary
        to remove craters that are less than ~5 pixels in diameter.
    amt : int
        Number of images to produce.
    zeropad : int
        Number of zeros to pad output file numbering.
    slivercut : float from 0 to 1
        Occasionally the code samples a small region near the pole,
        in which case the transformation from Plate Carree to
        Orthographic produces tiny slivers of the Moon surrounded
        by padding.  These images are useless, so the code trashes
        any images whose non-padding region has an width/height ratio
        less than slivercut.  Discarded images are not counted
        as part of amt.  Setting slivercut to 0 disables the cut.
        Setting it too close to 1 will lead to an infinite loop!
    outp : str or None
        If a string, will dump the long/lat boundary and crop
        bounds of all images to a pickle file.  File's name is
        obtained from outhead + outp.
    istart : int
        Output file starting number.  Useful for preventing overwriting
        of files when batch serializing the code (see __main__ script)
    seed : int or None
        np.random.seed input (for testing purposes).
    """

    # just in case we make this user-selectable later...
    origin = "upper"

    # If seed == None, uses an OS-dependent built-in
    # randomizer
    np.random.seed(seed)

    # Get craters
    AddPlateCarree_XY(craters, list(img.size), cdim=cdim, 
                      origin=origin)

    iglobe = ccrs.Globe(semimajor_axis=arad*1000., 
                    semiminor_axis=arad*1000.,
                    ellipse=None)

    # Determine log values of ilen range
    ilen_min = np.log10(ilen_range[0])
    ilen_max = np.log10(ilen_range[1])

    i = istart

    if outp:
        outpnames = []
        outpvals = []

    while i < istart + amt:

        # Determine image size to crop
        ilen = int(10**np.random.uniform(ilen_min, ilen_max))
        xm, ym = img.size[0] - ilen, img.size[1] - ilen
        xc = np.random.randint(0, xm)
        yc = np.random.randint(0, ym)
        box = [xc, yc, xc + ilen, yc + ilen]

        # Load necessary because crop may be a lazy operation
        # im.load() should copy it.
        # http://pillow.readthedocs.io/en/3.1.x/reference/Image.html
        im = img.crop(box)
        im.load()

        # Obtain long/lat bounds for coordinate transform
        ix = np.array([box[0], box[2]])
        iy = np.array([box[1], box[3]])
        llong, llat = pix2coord(ix, iy, cdim, list(img.size), origin=origin)
        llbd = np.r_[llong, llat[::-1]]

        # Downsample image
        im = im.resize([olen, olen])

        # Remove all craters that are too small to be seen in image
        ctr_sub = ResampleCraters(craters, llbd, im.size[1], arad=arad,
                                  minpix=minpix)

        # Convert Plate Carree to Orthographic
        [imgo, ctr_xy] = PlateCarree_to_Orthographic(im, None, llbd, ctr_sub, 
                                    iglobe=iglobe, ctr_sub=True, arad=arad,
                                    origin=origin, rgcoeff=1.2, dontsave=True,
                                    slivercut=slivercut)

        # If PlateCarree_to_Orthogonal returns NoneType, skip saving,
        # and don't add 1 to i
        if imgo is None:
            print("Discarding narrow image")
            continue

        # Output everything
        oname = outhead + "_{i:0{zp}d}".format(i=i, zp=zeropad)
        imgo.save(oname + ".png")
        ctr_xy.to_csv(oname + ".csv", index=False)

        # Add entry to outp
        if outp:
            outpnames.append(i)
            outpvals.append({"llbd": llbd, "box": box})

        i += 1

        # Create generator
        #yield [np.asanyarray(imgo), ctr_xy]

    pdict = dict( zip(outpnames, outpvals) )
    pickle.dump( pdict, open(outhead + outp, 'wb') )


def InitialImageCut(img, cdim, newcdim):
    """Crops image, so that the crop output
    can be used in GenDataset.

    Parameters
    ----------
    img : PIL.Image.Image
        Image
    cdim : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of image
    newcdim : list-like
        Crop boundaries (x_min, x_max, y_min, y_max).  There is
        currently NO CHECK that newcdim is within cdim!

    Returns
    -------
    img : PIL.Image.Image
        Cropped image
    """

    origin = "upper"

    x, y = coord2pix(np.array(newcdim[:2]), 
                    np.array(newcdim[2:]), cdim, img.size, 
                    origin=origin)

    # y is backward since origin is upper!
    box = [x[0], y[1], x[1], y[0]]
    img = img.crop(box)
    img.load()
    
    return img


# if __name__ == '__main__':

#     from mpi4py import MPI
#     import argparse

#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()

#     parser = argparse.ArgumentParser(description='Input data creation script.')
#     parser.add_argument('--image_path', metavar='imgpath', type=str, required=False,
#                         help='Path to the source image.', default="./LOLA_Global_20k.png")
#     parser.add_argument('--lu_csv_path', metavar='lupath', type=str, required=False,
#                         help='Path to LU78287 crater csv.', default="./LU78287GT.csv")
#     parser.add_argument('--alan_csv_path', metavar='alanpath', type=str, required=False,
#                         help='Path to LROC crater csv.', default="./LROCCraters.csv")
#     parser.add_argument('--outhead', metavar='outhead', type=str, required=False,
#                         help='Filepath and filename prefix of output files.', default="out/lola")
#     parser.add_argument('--amt', type=int, default=7500, required=False,
#                         help='Number of images each thread will make (multiply by number of \
#                         threads for total number of images produced).')
#     parser.add_argument('--cdim', nargs=4, type=int, required=False,
#                         help='[Min longitude, max, min latitude, max] of source image crop. \
#                         Crop creates global bounds for image set.')
#     parser.add_argument('--minpix', type=float, default=0., required=False,
#                         help='Minimum pixel diameter allowed in crater csv')
#     parser.add_argument('--slivercut', type=float, default=0.6, required=False,
#                         help='Minimum width/height aspect ratio to be acceptable image.')
#     args = parser.parse_args()

#     print("Thread {0} of {1}".format(rank, size))

#     img = Image.open(args.image_path).convert("L")
        
#     cdim = [-180, 180, -90, 90]
#     if args.cdim:
#         img = InitialImageCut(img, cdim, args.cdim)
#         cdim = args.cdim

#     craters = ReadCombinedCraterCSV(filealan=args.alan_csv_path, filelu=args.lu_csv_path,
#                                             dropfeatures=True)
#     # Co-opt ResampleCraters to remove all craters beyond subset cdim
#     # keep minpix = 0 (since we don't have pixel diameters yet)
#     craters = ResampleCraters(craters, cdim, None)

#     GenDataset(img, craters, args.outhead, ilen_range=np.array([600., 2000.]), arad=1737.4,
#                     olen=300, cdim=cdim, amt=args.amt, zeropad=5, minpix=args.minpix,
#                     slivercut=args.slivercut, outp="_p{0}.p".format(rank), 
#                     istart = rank*args.amt)

def gkern(l=5, sig=1.):
    """
    Creates Gaussian kernel with side length l and a sigma of sig
    """

    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.sum(kernel)


# https://stackoverflow.com/questions/10031580/how-to-write-simple-geometric-shapes-into-numpy-arrays
def circlemaker(r=10.):
    """
    Creates circle mask of radius r.
    """

    # mask grid extent (+1 to ensure we capture radius)
    rhext = int(r) + 1

    xx, yy = np.mgrid[-rhext:rhext + 1, -rhext:rhext + 1]
    circle = (xx**2 + yy**2) <= r**2

    return circle.astype(float)

# http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#circle
# Though that autodoc is terrible, and should be supplemented with
# http://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html
# and (file that defines static void Circle(...))
# https://github.com/opencv/opencv/blob/05b15943d6a42c99e5f921b7dbaa8323f3c042c6/modules/imgproc/src/drawing.cpp
def ringmaker(r=10., dr=1):
    """
    Creates ring of radius r and thickness dr.

    Parameters
    ----------
    r : float
        Ring radius
    dr : int
        Ring thickness (cv2.circle requires int)
    """

    # mask grid extent (dr/2 +1 to ensure we capture ring width
    # and radius); same philosophy as above
    rhext = int(np.ceil(r + dr/2.)) + 1

    # cv2.circle requires integer radius
    mask = np.zeros([2*rhext + 1, 2*rhext + 1], np.uint8)

    # Generate ring
    ring = cv2.circle(mask, (rhext,rhext), int(np.round(r)), 1, thickness=dr)

    return ring.astype(float)


def get_merge_indices(cen, imglen, ks_h, ker_shp):
    """Helper function that returns indices for merging 
    gaussian with base image, including edge case
    handling.  x and y are identical, so code
    is axis-neutral.

    Assumes INTEGER values for all inputs!
    """

    left = cen - ks_h; right = cen + ks_h + 1

    # Handle edge cases.
    # If left side of gaussian is beyond the left
    # end of the image.
    if left < 0:
        # Crop gaussian and shift image index
        # to lefthand side.
        img_l = 0; g_l = -left
    else:
        img_l = left; g_l = 0
    if right > imglen:
        img_r = imglen; g_r = ker_shp - (right - imglen)
    else:
        img_r = right; g_r = ker_shp

    return [img_l, img_r, g_l, g_r]


def make_density_map(craters, img, kernel=None, k_support = 8, k_sig=4., knn=10, 
                        beta=0.3, kdict={}, truncate=True):
    """Makes Gaussian kernel density maps.

    Parameters
    ----------
    craters : pandas.DataFrame
        craters dataframe that includes pixel x and y columns
    img : numpy.ndarray
        original image; assumes colour channel is last axis (tf standard)
    kernel : function, "knn" or None
        If a function is inputted, function must return an array of 
        length craters.shape[0].  If "knn",  uses variable kernel with 
            sigma = beta*<d_knn>,
        where <d_knn> is the mean Euclidean distance of the k = knn nearest 
        neighbouring craters.  If anything else is inputted, will use
        constant kernel size with sigma = k_sigma.
    k_support : int
        Kernel support (i.e. size of kernel stencil) coefficient.  Support
        is determined by kernel_support = k_support*sigma.  Defaults to 8.
    k_sig : float
        Sigma for constant sigma kernel.  Defaults to 1.
    knn : int
        k nearest neighbours, used for "knn" kernel.  Defaults to 10.
    beta : float
        Beta value used to calculate sigma for "knn" kernel.  Default 
        is 0.3.
    kdict : dict
        If kernel is custom function, dictionary of arguments passed to kernel.
    truncate : bool
        If True, truncate mask where image truncates
    """

    # Load blank density map
    imgshape = img.shape[:2]
    dmap = np.zeros(imgshape)

    # Get number of craters
    N_ctrs = craters.shape[0]

    # Obtain gaussian kernel sigma values
    # callable checks if kernel is function
    if callable(kernel):
        sigma = kernel(**kdict)
    # If knn is used
    elif kernel == "knn":
        # If we have more than 1 crater, select either nearest 11 or N_ctrs
        # neighbours, whichever is closer
        if N_ctrs > 1:
            kdt = kd(craters[["x","y"]].as_matrix(), leafsize=10)
            dnn = kdt.query(craters[["x","y"]].as_matrix(), \
                                    k=min(N_ctrs, knn + 1))[0][:, 1:].mean(axis=1)
        # Otherwise, assume there are craters "offscreen" half an image away
        else:
            dnn = 0.5*imgshape[0]*np.ones(1)
        sigma = beta*dnn
    else:
        sigma = k_sig*np.ones(N_ctrs)

    # Gaussian adding loop
    for i in range(N_ctrs):
        cx = int(craters["x"][i]); cy = int(craters["y"][i])

        # A bit convoluted, but ensures that kernel_support
        # is always odd so that centre of gaussian falls on
        # a pixel.
        ks_half = int( k_support*sigma[i] / 2)
        kernel_support = ks_half * 2 + 1
        kernel = gkern(kernel_support, sigma[i])

        # Calculate indices on image where kernel should be added
        [imxl, imxr, gxl, gxr] = get_merge_indices(cx, imgshape[1], 
                                                    ks_half, kernel_support)
        [imyl, imyr, gyl, gyr] = get_merge_indices(cy, imgshape[0], 
                                                    ks_half, kernel_support)

        # Add kernel to image
        dmap[imyl:imyr, imxl:imxr] += kernel[gyl:gyr, gxl:gxr]

    # Removes
    if truncate:
        if img.ndim == 3:
            dmap[img[:,:,0] == 0] = 0
        else:
            dmap[img == 0] = 0

    return dmap


def make_mask(craters, img, binary=True, rings=False, 
                                ringwidth=1, truncate=True):
    """Makes crater mask binary image (does not yet consider crater overlap).

    Parameters
    ----------
    craters : pandas.DataFrame
        craters dataframe that includes pixel x and y columns
    img : numpy.ndarray
        original image; assumes colour channel is last axis (tf standard)
    binary : bool
        If True, returns a binary image of crater masks
    rings : bool
        If True, mask uses hollow rings rather than filled circles
    ringwiddth : int
        If rings is True, ringwidth sets the width (dr) of the ring.
        
    truncate : bool
        If True, truncate mask where image truncates
    """

    # Load blank density map
    imgshape = img.shape[:2]
    dmap = np.zeros(imgshape)
    cx, cy = craters["x"].values.astype('int'), craters["y"].values.astype('int')
    radius = craters["Diameter (pix)"].values / 2.

    for i in range(craters.shape[0]):
        if rings:
            kernel = ringmaker(r=radius[i], dr=ringwidth)
        else:
            kernel = circlemaker(r=radius[i])
        # "Dummy values" so we can use get_merge_indices
        kernel_support = kernel.shape[0]
        ks_half = kernel_support // 2

        # Calculate indices on image where kernel should be added
        [imxl, imxr, gxl, gxr] = get_merge_indices(cx[i], imgshape[1],
                                                    ks_half, kernel_support)
        [imyl, imyr, gyl, gyr] = get_merge_indices(cy[i], imgshape[0],
                                                    ks_half, kernel_support)

        # Add kernel to image
        dmap[imyl:imyr, imxl:imxr] += kernel[gyl:gyr, gxl:gxr]
    
    if binary:
        dmap = (dmap > 0).astype(float)
    
    if truncate:
        if img.ndim == 3:
            dmap[img[:,:,0] == 0] = 0
        else:
            dmap[img == 0] = 0
    
    #add centroids to image
    #dmap[cy,cx] = 2

    return dmap
