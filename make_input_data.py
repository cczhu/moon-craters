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
import collections
import re
import cv2
import h5py
from scipy.spatial import cKDTree as kd

########## Read Cratering CSVs ##########

def ReadLROCCraterCSV(filename="./LROCCraters.csv", sortlat=True):
    """Reads LROC 5 - 20 km crater catalogue CSV.

    Parameters
    ----------
    filename : str, optional
        Filepath and name of LROC csv file.  Defaults to the one in the current
        folder.
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
    """Reads Head et al. 2010 >= 20 km diameter crater catalogue.

    Parameters
    ----------
    filename : str, optional
        Filepath and name of Head et al. csv file.  Defaults to the one in
        the current folder.
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
                                  filehead="./HeadCraters.csv",
                                  sortlat=True):
    """Combines LROC 5 - 20 km crater dataset with Head >= 20 km dataset.

    Parameters
    ----------
    filelroc : str, optional
        LROC crater file location.  Defaults to the one in the current folder.
    filehead : str, optional
        Head et al. crater file location.  Defaults to the one in the current
        folder.
    sortlat : bool, optional
        If `True` (default), order catalogue by latitude.

    Returns
    -------
    craters : pandas.DataFrame
        Craters data frame.
    """
    ctrs_head = ReadHeadCraterCSV(filename=filehead, sortlat=False)
    # Just in case.
    assert ctrs_head.shape == ctrs_head[ctrs_head["Diameter (km)"] > 20].shape
    ctrs_lroc = ReadLROCCraterCSV(filename=filelroc, sortlat=False)
    ctrs_lroc.drop(["tag"], axis=1, inplace=True)
    craters = pd.concat([ctrs_lroc, ctrs_head], axis=0, ignore_index=True,
                        copy=True)
    if sortlat:
        craters.sort_values(by='Lat', inplace=True)
    craters.reset_index(inplace=True, drop=True)

    return craters


def ReadSalamuniccarCraterCSV(filename="./LU78287GT.csv", dropfeatures=False,
                              sortlat=True):
    """Reads LU78287GT crater file CSV.  Also compatible with LU60645GT.

    Parameters
    ----------
    filename : str, optional
        csv file of craters.  Defaults to the one in the current folder.
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
    # Read in crater names.
    craters_names = ["Long", "Lat", "Radius (deg)", "Diameter (km)",
                     "D_range", "p", "Name"]
    craters_types = [float, float, float, float, float, int, str]
    craters = pd.read_csv(
        open(filename, 'r'), sep=',', usecols=list(range(1, 8)), header=0,
        engine="c", encoding="ISO-8859-1", names=craters_names,
        dtype=dict(zip(craters_names, craters_types)))

    # Truncate cyrillic characters.
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

    # String matching mini-function.
    def match_end(s):
        if re.match(r" [A-Z]", s[-2:]):
            return True
        return False

    # Find all crater names that ends with A - Z.
    basenames = (
        craters.loc[craters["Name"].notnull(), "Name"].apply(match_end))
    drop_index = basenames[basenames].index
    craters.drop(drop_index, inplace=True)


def ReadLROCLUCombinedCraterCSV(filelroc="./LROCCraters.csv",
                                filelu="./LU78287GT.csv",
                                dropfeatures=False, sortlat=True):
    """Combines LROC 5 - 20 km crater dataset with Goran Salamuniccar craters
    that are > 20 km.

    Parameters
    ----------
    filelroc : str, optional
        LROC crater file location.  Defaults to the one in the current folder.
    filelu : str, optional
        Salamuniccar crater file location.  Defaults to the one in the current
        folder.
    dropfeatures : bool, optional
        If `True` (defaults to `False`), drop satellite craters (those listed
        with "A", "B", "C"...), leaving only the whole crater (listed as "r").
        Only useful if you want to (crudely) remove secondary impacts.
    sortlat : bool, optional
        If `True` (default), order catalogue by latitude.

    Returns
    -------
    craters : pandas.DataFrame
        Craters data frame.
    """

    # Read in LU crater names.
    craters = ReadSalamuniccarCraterCSV(filename=filelu, sortlat=False,
                                        dropfeatures=dropfeatures)

    craters.drop(["Radius (deg)", "D_range", "p", "Name"],
                 axis=1, inplace=True)
    craters = craters[craters["Diameter (km)"] > 20]

    craters_lroc = pd.read_csv(filelroc, header=0, usecols=list(range(2, 5)))

    craters = pd.concat([craters, craters_lroc], axis=0, ignore_index=True,
                        copy=True)

    if sortlat:
        craters.sort_values(by='Lat', inplace=True)
    craters.reset_index(inplace=True, drop=True)

    return craters


def ReadMercuryCraterCSV(filename="./MercLargeCraters.csv", sortlat=True):
    """Reads crater file CSV from Fassett et al. 2011.

    Parameters
    ----------
    filename : str, optional
        csv file of craters.  Defaults to the one in the current folder.
    sortlat : bool, optional
        If `True` (default), order catalogue by latitude.

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
        Coordinate x.
    cy : float or ndarray
        Coordinate y.
    cdim : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of image.
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels.
    origin : 'upper' or 'lower', optional
        Based on imshow convention for displaying image y-axis. 'upper' means
        that [0, 0] is upper-left corner of image; 'lower' means it is
        bottom-left.

    Returns
    -------
    x : float or ndarray
        Pixel x positions.
    y : float or ndarray
        Pixel y positions.
    """

    x = imgdim[0] * (cx - cdim[0]) / (cdim[1] - cdim[0])

    if origin == "lower":
        y = imgdim[1] * (cy - cdim[2]) / (cdim[3] - cdim[2])
    else:
        y = imgdim[1] * (cdim[3] - cy) / (cdim[3] - cdim[2])

    return x, y


def pix2coord(x, y, cdim, imgdim, origin="upper"):
    """Converts image pixel locations to Plate Carree lat/long.  Assumes
    central meridian is at 0 (so long in [-180, 180)).

    Parameters
    ----------
    x : float or ndarray
        Pixel x positions.
    y : float or ndarray
        Pixel y positions.
    cdim : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of image.
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels.
    origin : 'upper' or 'lower', optional
        Based on imshow convention for displaying image y-axis. 'upper' means
        that [0, 0] is upper-left corner of image; 'lower' means it is
        bottom-left.

    Returns
    -------
    cx : float or ndarray
        Coordinate x.
    cy : float or ndarray
        Coordinate y.
    """

    cx = (x / imgdim[0]) * (cdim[1] - cdim[0]) + cdim[0]

    if origin == "lower":
        cy = (y / imgdim[1]) * (cdim[3] - cdim[2]) + cdim[2]
    else:
        cy = cdim[3] - (y / imgdim[1]) * (cdim[3] - cdim[2])

    return cx, cy

########## Metres to pixels conversion ##########

def km2pix(imgheight, latextent, dc=1., a=1737.4):
    """Returns conversion from km to pixels.

    Parameters
    ----------
    imgheight : float
        Height of image in pixels.
    latextent : float
        Latitude extent of image in degrees.
    dc : float from 0 to 1, optional
        Scaling factor for distortions.
    a : float, optional
        World radius in km.  Default is Moon (1737.4 km).

    Returns
    -------
    km2pix : float
        Conversion factor pix/km
    """
    return (180. / np.pi) * imgheight * dc / latextent / a

########## Warp Images and CSVs ##########

def regrid_shape_aspect(regrid_shape, target_extent):
    """Helper function copied from cartopy.img_transform for resizing an image
    without changing its aspect ratio.

    Parameters
    ----------
    regrid_shape : int or float
        Target length of the shorter axis (in units of pixels).
    target_extent : some
        Width and height of the target image (generally not in units of
        pixels).

    Returns
    -------
    regrid_shape : tuple
        Width and height of the target image in pixels.
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
    """Warps images with cartopy.img_transform.warp_array, then plots them with
    imshow.  Based on cartopy.mpl.geoaxes.imshow.

    Parameters
    ----------
    img : numpy.ndarray
        Image as a 2D array.
    iproj : cartopy.crs.Projection instance
        Input coordinate system.
    iextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of input.
    oproj : cartopy.crs.Projection instance
        Output coordinate system.
    oextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of output.
    origin : "lower" or "upper", optional
        Based on imshow convention for displaying image y-axis.  "upper" means
        [0,0] is in the upper-left corner of the image; "lower" means it's in
        the bottom-left.
    rgcoeff : float, optional
        Fractional size increase of transformed image height.  Generically set
        to 1.2 to prevent loss of fidelity during transform (though some of it
        is inevitably lost due to warping).
    """

    if iproj == oproj:
        raise Warning("Input and output transforms are identical!"
                      "Returing input!")
        return img

    if origin == 'upper':
        # Regridding operation implicitly assumes origin of image is
        # 'lower', so adjust for that here.
        img = img[::-1]

    # rgcoeff is padding when we rescale the image later.
    regrid_shape = rgcoeff * min(img.shape)
    regrid_shape = regrid_shape_aspect(regrid_shape,
                                       oextent)

    # cimg.warp_array uses cimg.mesh_projection, which cannot handle any
    # zeros being used in iextent.  Create iextent_nozeros to fix.
    iextent_nozeros = np.array(iextent)
    iextent_nozeros[iextent_nozeros == 0] = 1e-8
    iextent_nozeros = list(iextent_nozeros)

    imgout, extent = cimg.warp_array(img,
                                     source_proj=iproj,
                                     source_extent=iextent_nozeros,
                                     target_proj=oproj,
                                     target_res=regrid_shape,
                                     target_extent=oextent,
                                     mask_extrapolated=True)

    if origin == 'upper':
        # Transform back.
        imgout = imgout[::-1]

    return imgout


def WarpImagePad(img, iproj, iextent, oproj, oextent, origin="upper",
                 rgcoeff=1.2, fillbg="black", original_dims=True):
    """Wrapper for WarpImage that adds padding to warped image to make it the
    same size as the original.

    Parameters
    ----------
    img : numpy.ndarray
        Image as a 2D array.
    iproj : cartopy.crs.Projection instance
        Input coordinate system.
    iextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of input.
    oproj : cartopy.crs.Projection instance
        Output coordinate system.
    oextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of output.
    origin : "lower" or "upper", optional
        Based on imshow convention for displaying image y-axis.  "upper" means
        [0,0] is in the upper-left corner of the image; "lower" means it's in
        the bottom-left.
    rgcoeff : float, optional
        Fractional size increase of transformed image height.  Generically set
        to 1.2 to prevent loss of fidelity during transform (though some of it
        is inevitably lost due to warping).
    fillbg : 'black' or 'white', optional.
        Fills padding with either black (0) or white (255) values.  Default is
        black.
    original_dims : bool, optional
        If True, rescale and add additional padding to keep exact dimensions of
        original image.  Warped image will be sized to fit the original
        image's smaller axis.

    Returns
    -------
    If original_dims == True:
    imgo : PIL.Image.Image
        Warped image with padding, resized to the original image's dimensions.
    imgw.size : tuple
        Width, height of picture without padding
    offset : tuple
        Pixel width of (left, top)-side padding
    If original_dims == False:
    imgw : PIL.Image.Image
        Warped image with padding.
    """
    # Based off of <https://stackoverflow.com/questions/2563822/
    # how-do-you-composite-an-image-onto-another-image-with-pil-in-python>

    if type(img) == Image.Image:
        img = np.asanyarray(img)

    # Check that we haven't been given a corrupted image.
    assert img.sum() > 0, "Image input to WarpImagePad is blank!"

    # Set background colour
    if fillbg == "white":
        bgval = 255
    else:
        bgval = 0

    # Warp image.
    imgw = WarpImage(img, iproj, iextent, oproj, oextent,
                     origin=origin, rgcoeff=rgcoeff)

    # Remove mask, turn image into Image.Image.
    imgw = np.ma.filled(imgw, fill_value=bgval)
    imgw = Image.fromarray(imgw, mode="L")

    if original_dims:
        # Resize to height of original, maintaining aspect ratio.  Note
        # img.shape = height, width, and imgw.size and
        # imgo.size = width, height.
        imgw_loh = imgw.size[0] / imgw.size[1]

        # If imgw is stretched horizontally compared to img.
        if imgw_loh > (img.shape[1] / img.shape[0]):
            imgw = imgw.resize([img.shape[0],
                                int(np.round(img.shape[0] / imgw_loh))],
                               resample=Image.NEAREST)
        # If imgw is stretched vertically.
        else:
            imgw = imgw.resize([int(np.round(imgw_loh * img.shape[0])),
                                img.shape[0]], resample=Image.NEAREST)

        # Make background image and paste two together.
        imgo = Image.new('L', (img.shape[1], img.shape[0]), (bgval))
        offset = ((imgo.size[0] - imgw.size[0]) // 2,
                  (imgo.size[1] - imgw.size[1]) // 2)
        imgo.paste(imgw, offset)

        return imgo, imgw.size, offset

    return imgw


def WarpCraterLoc(craters, geoproj, oproj, oextent, imgdim, llbd=None,
                  origin="upper"):
    """Wrapper for WarpImage that adds padding to warped image to make it the
    same size as the original.

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
        ctr_xlim = ((craters["Long"] >= llbd[0]) &
                    (craters["Long"] <= llbd[1]))
        ctr_ylim = ((craters["Lat"] >= llbd[2]) &
                    (craters["Lat"] <= llbd[3]))
        ctr_wrp = craters.loc[ctr_xlim & ctr_ylim, :].copy()

    # Get output projection coords.
    # [:,:2] becaus we don't need elevation data
    # If statement is in case ctr_wrp has nothing in it
    if ctr_wrp.shape[0]:
        ilong = ctr_wrp["Long"].as_matrix()
        ilat = ctr_wrp["Lat"].as_matrix()
        res = oproj.transform_points(x=ilong, y=ilat,
                                     src_crs=geoproj)[:, :2]

        # Get output
        ctr_wrp["x"], ctr_wrp["y"] = coord2pix(res[:, 0], res[:, 1],
                                               oextent, imgdim, origin=origin)
    else:
        ctr_wrp["x"] = []
        ctr_wrp["y"] = []

    return ctr_wrp

########## Cropping Helper Functions ##########

def is_crater_out_of_box(ctr, box=[0, 0, 0, 0], pad=2):
    """ Determine if a crater's rim falls within a box.

    Helper function to PlateCarree_to_Orthographic_Cropped for determining
    whether to discard crater data when cropping out the central region of the
    transformed image.
    """
    # Integer floor of (crater radius / sqrt(2)) - padding.
    ctr_rad_sqrt2_pad = int(ctr['Diameter (pix)'] / (2. * np.sqrt(2))) - pad
    # Check if a crater's centre falls beyond the box of size
    #     [box[0] - ctr_rad_sqrt2_pad, box[1] - ctr_rad_sqrt2_pad,
    #      box[2] + ctr_rad_sqrt2_pad, box[3] + ctr_rad_sqrt2_pad]
    # which approximates the furthest a circle's centre can get from the box
    # while part of it still falls within the box.  Can do a more accurate cut
    # with a slightly larger box that had rounded corners, but that's a lot of
    # effort and computation for little gain.
    if ((ctr['x'] > box[0] - ctr_rad_sqrt2_pad) &
            (ctr['x'] < box[2] + ctr_rad_sqrt2_pad) &
            (ctr['y'] > box[1] - ctr_rad_sqrt2_pad) &
            (ctr['y'] < box[3] + ctr_rad_sqrt2_pad)):
        return False
    return True

############# Warp Plate Carree to Orthographic ###############

def PlateCarree_to_Orthographic(img, oname, llbd, craters, iglobe=None,
                                ctr_sub=False, arad=1737.4, origin="upper",
                                rgcoeff=1.2, slivercut=0.):
    """Transform Plate Carree image and associated csv file into Orthographic.

    Parameters
    ----------
    img : PIL.Image.image or str
        File or filename.
    oname : str
        Output filename.
    llbd : list-like
        Long/lat limits (long_min, long_max, lat_min, lat_max) of image.
    craters : pandas.DataFrame
        Craters catalogue.
    iglobe : cartopy.crs.Geodetic instance
        Globe for images.  If False, defaults to spherical Moon.
    ctr_sub : bool, optional
        If `True`, assumes craters dataframe includes only craters within
        image. If `False` (default_, llbd used to cut craters from outside
        image out of (copy of) dataframe.
    arad : float
        World radius in km.  Default is Moon (1737.4 km).
    origin : "lower" or "upper", optional
        Based on imshow convention for displaying image y-axis.  "upper"
        (default) means that [0,0] is upper-left corner of image; "lower" means
        it is bottom-left.
    rgcoeff : float, optional
        Fractional size increase of transformed image height.  By default set
        to 1.2 to prevent loss of fidelity during transform (though warping can
        be so extreme that this might be meaningless).
    slivercut : float from 0 to 1, optional
        If transformed image aspect ratio is too narrow (and would lead to a
        lot of padding, return null images).

    Returns
    -------
    imgo : PIL.Image.image
        Transformed, padded image in PIL.Image format.
    ctr_xy : pandas.DataFrame
        Craters with transformed x, y pixel positions and pixel radii.
    distortion_coefficient : float
        Ratio between the central heights of the transformed image and original
        image.
    """

    # If user doesn't provide Moon globe properties.
    if not iglobe:
        iglobe = ccrs.Globe(semimajor_axis=arad*1000.,
                            semiminor_axis=arad*1000., ellipse=None)

    # Set up Geodetic (long/lat), Plate Carree (usually long/lat, but not when
    # globe != WGS84) and Orthographic projections.
    geoproj = ccrs.Geodetic(globe=iglobe)
    iproj = ccrs.PlateCarree(globe=iglobe)
    oproj = ccrs.Orthographic(central_longitude=np.mean(llbd[:2]),
                              central_latitude=np.mean(llbd[2:]),
                              globe=iglobe)

    # Create and transform coordinates of image corners and edge midpoints.
    # Due to Plate Carree and Orthographic's symmetries, max/min x/y values of
    # these 9 points represent extrema of the transformed image.
    xll = np.array([llbd[0], np.mean(llbd[:2]), llbd[1]])
    yll = np.array([llbd[2], np.mean(llbd[2:]), llbd[3]])
    xll, yll = np.meshgrid(xll, yll)
    xll = xll.ravel()
    yll = yll.ravel()

    # [:,:2] because we don't need elevation data.
    res = iproj.transform_points(x=xll, y=yll, src_crs=geoproj)[:, :2]
    iextent = [min(res[:, 0]), max(res[:, 0]), min(res[:, 1]), max(res[:, 1])]

    res = oproj.transform_points(x=xll, y=yll, src_crs=geoproj)[:, :2]
    oextent = [min(res[:, 0]), max(res[:, 0]), min(res[:, 1]), max(res[:, 1])]

    # Sanity check for narrow images; done before the most expensive part of
    # the function.
    oaspect = (oextent[1] - oextent[0]) / (oextent[3] - oextent[2])
    if oaspect < slivercut:
        return [None, None]

    if type(img) != Image.Image:
        img = Image.open(img).convert("L")

    # Warp image.
    imgo, imgwshp, offset = WarpImagePad(img, iproj, iextent, oproj, oextent,
                                         origin=origin, rgcoeff=rgcoeff,
                                         fillbg="black", original_dims=True)

    # Convert crater x, y position.
    if ctr_sub:
        llbd_in = None
    else:
        llbd_in = llbd
    ctr_xy = WarpCraterLoc(craters, geoproj, oproj, oextent, imgwshp,
                           llbd=llbd_in, origin=origin)
    # Shift crater x, y positions by offset (origin doesn't matter for y-shift,
    # since padding is symmetric).
    ctr_xy.loc[:, "x"] += offset[0]
    ctr_xy.loc[:, "y"] += offset[1]

    # Pixel scale for orthographic determined (for images small enough that
    # tan(x) approximately equals x + 1/3x^3 + ...) by l = R_moon*theta,
    # where theta is the latitude extent of the centre of the image.  Because
    # projection transform doesn't guarantee central vertical axis will keep
    # its pixel resolution, we need to calculate the conversion coefficient
    #   C = (res[7,1]- res[1,1])/(oextent[3] - oextent[2]).
    #   C0*pix height/C = theta
    # Where theta is the latitude extent and C0 is the theta per pixel
    # conversion for the Plate Carree image).  Thus
    #   l_ctr = R_moon*C0*pix_ctr/C.
    distortion_coefficient = ((res[7, 1] - res[1, 1]) /
                              (oextent[3] - oextent[2]))
    if distortion_coefficient < 0.7:
        raise ValueError("Distortion Coefficient cannot be"
                         " {0:.2f}!".format(distortion_coefficient))
    pixperkm = km2pix(imgo.size[1], llbd[3] - llbd[2],
                      dc=distortion_coefficient, a=arad)
    ctr_xy["Diameter (pix)"] = ctr_xy["Diameter (km)"] * pixperkm

    return [imgo, ctr_xy, distortion_coefficient]


def PlateCarree_to_Orthographic_Cropped(img, ilen, llbd, craters, iglobe=None,
                                        arad=1737.4, origin="upper",
                                        rgcoeff=1.3, paderr=0.7):
    """Transform Plate Carree image and associated csv file into Orthographic.

    Parameters
    ----------
    img : PIL.Image.image
        Image file.
    ilen : int
        Size of central region of image to crop.
    llbd : list-like
        Long/lat limits (long_min, long_max, lat_min, lat_max) of image.
    craters : pandas.DataFrame
        Craters catalogue.  To minimize cropping time, include only craters
        within larger image.
    iglobe : cartopy.crs.Geodetic instance
        Globe for images.  If False, defaults to spherical Moon.
    arad : float
        World radius in km.  Default is Moon (1737.4 km).
    origin : "lower" or "upper", optional
        Based on imshow convention for displaying image y-axis.  "upper"
        (default) means that [0,0] is upper-left corner of image; "lower" means
        it is bottom-left.
    rgcoeff : float, optional
        Fractional size increase of transformed image height.  By default set
        to 1.2 to prevent loss of fidelity during transform (though warping can
        be so extreme that this might be meaningless).
    paderr : float from 0 to 1, optional
        If not zero, throw an exception if the transformed image aspect ratio
        is below this value.

    Returns
    -------
    imgo : PIL.Image.image
        Transformed, padded image in PIL.Image format.
    ctr_xy : pandas.DataFrame
        Craters with transformed x, y pixel positions and pixel radii.
    pixperkm : float
        Conversion factor between km and image pixel.
    imgo_uc_size : tuple
        Size of imgo before it was cropped.
    centrallonglat_xy : pandas.DataFrame
        Central longitude and latitude, transformed to x, y pixel coords.
    """

    # If user doesn't provide Moon globe properties.
    if not iglobe:
        iglobe = ccrs.Globe(semimajor_axis=arad*1000.,
                            semiminor_axis=arad*1000., ellipse=None)

    # Set up Geodetic (long/lat), Plate Carree (usually long/lat, but not when
    # globe != WGS84) and Orthographic projections.
    geoproj = ccrs.Geodetic(globe=iglobe)
    iproj = ccrs.PlateCarree(globe=iglobe)
    oproj = ccrs.Orthographic(central_longitude=np.mean(llbd[:2]),
                              central_latitude=np.mean(llbd[2:]),
                              globe=iglobe)

    # Create and transform coordinates of image corners and edge midpoints.
    # Due to Plate Carree and Orthographic's symmetries, max/min x/y values of
    # these 9 points represent extrema of the transformed image.
    xll = np.array([llbd[0], np.mean(llbd[:2]), llbd[1]])
    yll = np.array([llbd[2], np.mean(llbd[2:]), llbd[3]])
    xll, yll = np.meshgrid(xll, yll)
    xll = xll.ravel()
    yll = yll.ravel()

    # [:,:2] because we don't need elevation data.
    res = iproj.transform_points(x=xll, y=yll, src_crs=geoproj)[:, :2]
    iextent = [min(res[:, 0]), max(res[:, 0]), min(res[:, 1]), max(res[:, 1])]

    res = oproj.transform_points(x=xll, y=yll, src_crs=geoproj)[:, :2]
    oextent = [min(res[:, 0]), max(res[:, 0]), min(res[:, 1]), max(res[:, 1])]

    # Sanity check for narrow images; done before the most expensive part of
    # the function.
    oaspect = (oextent[1] - oextent[0]) / (oextent[3] - oextent[2])
    if oaspect < paderr:
        raise ValueError("Aspect ratio {0} has fallen below {1}!  Try "
                         "elongating the input image.".format(oaspect, paderr))

    # Transform the image from Plate Carree to orthographic.
    imgo = WarpImagePad(img, iproj, iextent, oproj, oextent, origin=origin,
                        rgcoeff=rgcoeff, fillbg="black", original_dims=False)

    # The projection transform doesn't guarantee the central vertical axis of
    # the new image will be free of padding (thus throwing off pixel-to-km
    # scaling relations). Calculate the distortion_coefficient, the fraction of
    # the central vertical axis that is not padding.
    distortion_coefficient = ((res[7, 1] - res[1, 1]) /
                              (oextent[3] - oextent[2]))
    if distortion_coefficient < 0.7:
        raise ValueError("Distortion coefficient {0} has fallen below {1}!"
                         "Try reducing input image elongation, or using a"
                         "smaller overall image".format(
                             distortion_coefficient, 0.7))

    # Resize the image (including correcting for central vertical axis
    # padding).
    corrected_height = np.round(img.size[1] /
                                distortion_coefficient).astype(int)
    imgo = imgo.resize([np.round(imgo.size[0] / imgo.size[1] *
                                 corrected_height).astype(int),
                        corrected_height], resample=Image.NEAREST)

    # Convert crater x, y position.
    ctr_xy = WarpCraterLoc(craters, geoproj, oproj, oextent, imgo.size,
                           llbd=None, origin=origin)
    # Calculate crater pixel diameters.  Even though we've used
    # distortion_correction to enlarge the image, we still need to pass
    # distortion correction into km2pix.
    pixperkm = km2pix(imgo.size[1], llbd[3] - llbd[2],
                      dc=distortion_coefficient, a=arad)
    ctr_xy["Diameter (pix)"] = ctr_xy["Diameter (km)"] * pixperkm

    # Determine x, y position of central lat/long.
    centrallonglat = pd.DataFrame({"Long": [xll[4]], "Lat": [yll[4]]})
    centrallonglat_xy = WarpCraterLoc(centrallonglat, geoproj, oproj, oextent,
                                      imgo.size, llbd=None, origin=origin)

    # Determine x, y offset for central region of image.
    offset = [(imgo.size[0] - ilen) // 2, (imgo.size[1] - ilen) // 2]
    cropbox = [offset[0], offset[1], offset[0] + ilen, offset[1] + ilen]

    # Crop central image
    imgo_crop = imgo.crop(cropbox)

    drop_craters = ctr_xy.apply(is_crater_out_of_box, axis=1,
                                box=cropbox, pad=2)
    drop_index = drop_craters[drop_craters].index
    ctr_xy.drop(drop_index, inplace=True)

    # Shift crater x, y positions by offset (origin doesn't matter for y-shift,
    # since padding is symmetric).
    ctr_xy.loc[:, "x"] -= offset[0]
    ctr_xy.loc[:, "y"] -= offset[1]

    # Shift central long/lat
    centrallonglat_xy.loc[:, "x"] -= offset[0]
    centrallonglat_xy.loc[:, "y"] -= offset[1]

    return [imgo_crop, ctr_xy, pixperkm, imgo.size, centrallonglat_xy]

############# Create target dataset (and helper functions) #############

def gkern(sl=5, sig=1.):
    """Creates Gaussian kernel with side length l and a sigma of sig.
    """

    ax = np.arange(-sl // 2 + 1., sl // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.sum(kernel)


def circlemaker(r=10.):
    """
    Creates circle mask of radius r.
    """
    # Based on <https://stackoverflow.com/questions/10031580/
    # how-to-write-simple-geometric-shapes-into-numpy-arrays>

    # Mask grid extent (+1 to ensure we capture radius).
    rhext = int(r) + 1

    xx, yy = np.mgrid[-rhext:rhext + 1, -rhext:rhext + 1]
    circle = (xx**2 + yy**2) <= r**2

    return circle.astype(float)


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
    # See <http://docs.opencv.org/2.4/modules/core/doc/
    # drawing_functions.html#circle>, supplemented by
    # <http://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html>
    # and <https://github.com/opencv/opencv/blob/
    # 05b15943d6a42c99e5f921b7dbaa8323f3c042c6/modules/imgproc/
    # src/drawing.cpp>.

    # mask grid extent (dr/2 +1 to ensure we capture ring width
    # and radius); same philosophy as above
    rhext = int(np.ceil(r + dr / 2.)) + 1

    # cv2.circle requires integer radius
    mask = np.zeros([2 * rhext + 1, 2 * rhext + 1], np.uint8)

    # Generate ring
    ring = cv2.circle(mask, (rhext, rhext), int(np.round(r)), 1, thickness=dr)

    return ring.astype(float)


def get_merge_indices(cen, imglen, ks_h, ker_shp):
    """Helper function that returns indices for merging gaussian with base
    image, including edge case handling.  x and y are identical, so code is
    axis-neutral.

    Assumes INTEGER values for all inputs!
    """

    left = cen - ks_h
    right = cen + ks_h + 1

    # Handle edge cases.  If left side of gaussian is beyond the left end of
    # the image, for example, crop gaussian and shift image index to lefthand
    # side.
    if left < 0:
        img_l = 0
        g_l = -left
    else:
        img_l = left
        g_l = 0
    if right > imglen:
        img_r = imglen
        g_r = ker_shp - (right - imglen)
    else:
        img_r = right
        g_r = ker_shp

    return [img_l, img_r, g_l, g_r]


def make_density_map(craters, img, kernel=None, k_support=8, k_sig=4.,
                     knn=10, beta=0.3, kdict={}, truncate=True):
    """Makes Gaussian kernel density maps.

    Parameters
    ----------
    craters : pandas.DataFrame
        craters dataframe that includes pixel x and y columns
    img : numpy.ndarray
        original image; assumes colour channel is last axis (tf standard)
    kernel : function, "knn" or None, optional
        If a function is inputted, function must return an array of
        length craters.shape[0].  If "knn",  uses variable kernel with
            sigma = beta*<d_knn>,
        where <d_knn> is the mean Euclidean distance of the k = knn nearest
        neighbouring craters.  If anything else is inputted, will use
        constant kernel size with sigma = k_sigma.
    k_support : int, optional
        Kernel support (i.e. size of kernel stencil) coefficient.  Support
        is determined by kernel_support = k_support*sigma.  Defaults to 8.
    k_sig : float, optional
        Sigma for constant sigma kernel.  Defaults to 1.
    knn : int, optional
        k nearest neighbours, used for "knn" kernel.  Defaults to 10.
    beta : float, optional
        Beta value used to calculate sigma for "knn" kernel.  Default
        is 0.3.
    kdict : dict, optional
        If kernel is custom function, dictionary of arguments passed to kernel.
    truncate : bool, optional
        If `True` (default), truncate mask where image truncates
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
            kdt = kd(craters[["x", "y"]].as_matrix(), leafsize=10)
            dnn = kdt.query(craters[["x", "y"]].as_matrix(),
                            k=min(N_ctrs, knn + 1))[0][:, 1:].mean(axis=1)
        # Otherwise, assume there are craters "offscreen" half an image away
        else:
            dnn = 0.5 * imgshape[0] * np.ones(1)
        sigma = beta * dnn
    else:
        sigma = k_sig * np.ones(N_ctrs)

    # Gaussian adding loop
    for i in range(N_ctrs):
        cx = int(craters["x"][i])
        cy = int(craters["y"][i])

        # A bit convoluted, but ensures that kernel_support is always odd so
        # that centre of gaussian falls on a pixel.
        ks_half = int(k_support * sigma[i] / 2)
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
            dmap[img[:, :, 0] == 0] = 0
        else:
            dmap[img == 0] = 0

    return dmap


def make_mask(craters, img, binary=True, rings=False, ringwidth=1,
              truncate=True):
    """Makes crater mask binary image (does not yet consider crater overlap).

    Parameters
    ----------
    craters : pandas.DataFrame
        Craters catalogue that includes pixel x and y columns.
    img : numpy.ndarray
        Original image; assumes colour channel is last axis (tf standard).
    binary : bool, optional
        If True, returns a binary image of crater masks.
    rings : bool, optional
        If True, mask uses hollow rings rather than filled circles.
    ringwidth : int, optional
        If rings is True, ringwidth sets the width (dr) of the ring.
    truncate : bool
        If True, truncate mask where image truncates.

    Returns
    -------
    mask : numpy.ndarray
        Target mask image.
    """

    # Load blank density map
    imgshape = img.shape[:2]
    mask = np.zeros(imgshape)
    cx = craters["x"].values.astype('int')
    cy = craters["y"].values.astype('int')
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
        mask[imyl:imyr, imxl:imxr] += kernel[gyl:gyr, gxl:gxr]

    if binary:
        mask = (mask > 0).astype(float)

    if truncate:
        if img.ndim == 3:
            mask[img[:, :, 0] == 0] = 0
        else:
            mask[img == 0] = 0

    return mask

############# Create dataset (and helper functions) #############

def AddPlateCarree_XY(craters, imgdim, cdim=[-180., 180., -90., 90.], 
                      origin="upper"):
    """Adds x and y pixel locations to craters dataframe.

    Parameters
    ----------
    craters : pandas.DataFrame
        Crater info
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels
    cdim : list-like, optional
        Coordinate limits (x_min, x_max, y_min, y_max) of image.  Default is
        [-180., 180., -90., 90.].
    origin : "upper" or "lower", optional
        Based on imshow convention for displaying image y-axis.
        "upper" means that [0,0] is upper-left corner of image;
        "lower" means it is bottom-left.
    """
    x, y = coord2pix(craters["Long"].as_matrix(), craters["Lat"].as_matrix(),
                     cdim, imgdim, origin=origin)
    craters["x"] = x
    craters["y"] = y


def ResampleCraters(craters, llbd, imgheight, arad=1737.4, minpix=0):
    """Crops crater file, and removes craters smaller than some minimum value.

    Parameters
    ----------
    craters : pandas.DataFrame
        Crater dataframe.
    llbd : list-like
        Long/lat limits (long_min, long_max, lat_min, lat_max) of image.
    imgheight : int
        Pixel height of image.
    arad : float, optional
        World radius in km.  Defaults to Moon radius (1737.4 km).
    minpix : int, optional
        Minimium crater pixel size to be included in output.  Default is 0
        (equvalent to no cutoff).

    Returns
    -------
    ctr_sub : pandas.DataFrame
        Cropped and filtered dataframe.
    """

    # Get subset of craters within llbd limits.
    ctr_xlim = (craters["Long"] >= llbd[0]) & (craters["Long"] <= llbd[1])
    ctr_ylim = (craters["Lat"] >= llbd[2]) & (craters["Lat"] <= llbd[3])
    ctr_sub = craters.loc[ctr_xlim & ctr_ylim, :].copy()

    if minpix > 0:
        # Obtain pixel per km conversion factor.  Use latitude because Plate
        # Carree doesn't distort along this axis.
        pixperkm = km2pix(imgheight, llbd[3] - llbd[2], dc=1., a=arad)
        minkm = minpix / pixperkm

        # Remove craters smaller than pixel limit.
        ctr_sub = ctr_sub[ctr_sub["Diameter (km)"] >= minkm]
        ctr_sub.reset_index(inplace=True, drop=True)

    return ctr_sub


def InitialImageCut(img, cdim, newcdim):
    """Crops image, so that the crop output can be used in GenDataset.

    Parameters
    ----------
    img : PIL.Image.Image
        Image
    cdim : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of image.
    newcdim : list-like
        Crop boundaries (x_min, x_max, y_min, y_max).  There is
        currently NO CHECK that newcdim is within cdim!

    Returns
    -------
    img : PIL.Image.Image
        Cropped image
    """
    x, y = coord2pix(np.array(newcdim[:2]), np.array(newcdim[2:]), cdim,
                     img.size, origin="upper")

    # y is backward since origin is upper!
    box = [x[0], y[1], x[1], y[0]]
    img = img.crop(box)
    img.load()

    return img


def GenDataset(img, craters, outhead, rawlen_range=[1000, 2000],
               rawlen_dist='log', ilen=256, cdim=[-180., 180., -60., 60.],
               arad=1737.4, minpix=0, tglen=256, binary=True, rings=True,
               ringwidth=1, truncate=True, amt=100, istart=0, seed=None,
               verbose=False):
    """Generates random dataset from a global DEM and crater catalogue.

    The function randomly samples small images from a global digital elevation
    map (DEM) that uses a Plate Carree projection, and converts the small
    images to Orthographic projection.  Pixel coordinates and radii of craters
    from the catalogue that fall within each image are placed in a
    corresponding Pandas dataframe.  Images and dataframes are saved to disk in
    hdf5 format.

    Parameters
    ----------
    img : PIL.Image.Image
        Source image.
    craters : pandas.DataFrame
        Crater catalogue .csv.
    outhead : str
        Filepath and file prefix of the image and crater table hdf5 files.
    rawlen_range : list-like, optional
        Lower and upper bounds of raw image widths, in pixels, to crop from
        source.  To always crop the same sized image, set lower bound to the
        same value as the upper.  Default is [300, 4000].
    rawlen_dist : 'uniform' or 'log'
        Distribution from which to randomly sample image widths.  'uniform' is
        uniform sampling, and 'log' is loguniform sampling.
    ilen : int, optional
        Input image width, in pixels.  Cropped images will be downsampled to
        this size.  Default is 256.
    cdim : list-like, optional
        Coordinate limits (x_min, x_max, y_min, y_max) of image.  Default is
        LRO-Kaguya's [-180., 180., -60., 60.].
    arad : float. optional
        World radius in km.  Defaults to Moon radius (1737.4 km).
    minpix : int, optional
        Minimum crater diameter in pixels to be included in crater list.
        Useful when the smallest craters in the catalogue are smaller than 1
        pixel in diameter.
    tglen : int, optional
        Target image width, in pixels.
    binary : bool, optional
        If True, returns a binary image of crater masks.
    rings : bool, optional
        If True, mask uses hollow rings rather than filled circles.
    ringwidth : int, optional
        If rings is True, ringwidth sets the width (dr) of the ring.
    truncate : bool
        If True, truncate mask where image truncates.
    amt : int, optional
        Number of images to produce.  100 by default.
    istart : int
        Output file starting number, when creating datasets spanning multiple
        files.
    seed : int or None
        np.random.seed input (for testing purposes).
    verbose : bool
        If True, prints out number of image being generated.
    """

    # just in case we ever make this user-selectable...
    origin = "upper"

    # Seed random number generator.
    np.random.seed(seed)

    # Get craters.
    AddPlateCarree_XY(craters, list(img.size), cdim=cdim, origin=origin)

    iglobe = ccrs.Globe(semimajor_axis=arad*1000., semiminor_axis=arad*1000.,
                        ellipse=None)

    # Create random sampler (either uniform or loguniform).
    if rawlen_dist == 'log':
        rawlen_min = np.log10(rawlen_range[0])
        rawlen_max = np.log10(rawlen_range[1])

        def random_sampler():
            return int(10**np.random.uniform(rawlen_min, rawlen_max))
    else:

        def random_sampler():
            return np.random.randint(rawlen_range[0], rawlen_range[1] + 1)

    # Initialize output hdf5s.
    imgs_h5 = h5py.File(outhead + '_images.hdf5', 'w')
    imgs_h5_inputs = imgs_h5.create_dataset("input_images", (amt, ilen, ilen),
                                            dtype='uint8')
    imgs_h5_inputs.attrs['definition'] = "Input image dataset."
    imgs_h5_tgts = imgs_h5.create_dataset("target_masks", (amt, tglen, tglen),
                                          dtype='float32')
    imgs_h5_tgts.attrs['definition'] = "Target mask dataset."
    imgs_h5_llbd = imgs_h5.create_group("longlat_bounds")
    imgs_h5_llbd.attrs['definition'] = ("(long min, long max, lat min, "
                                        "lat max) of the cropped image.")
    imgs_h5_box = imgs_h5.create_group("pix_bounds")
    imgs_h5_box.attrs['definition'] = ("Pixel bounds of the Global DEM region"
                                       " that was cropped for the image.")
    imgs_h5_dc = imgs_h5.create_group("pix_distortion_coefficient")
    imgs_h5_dc.attrs['definition'] = ("Distortion coefficient due to "
                                      "projection transformation.")
    craters_h5 = pd.HDFStore(outhead + '_craters.hdf5', 'w')

    # Zero-padding for hdf5 keys.
    zeropad = int(np.log10(amt)) + 1

    for i in range(amt):

        # Current image number.
        img_number = "img_{i:0{zp}d}".format(i=istart + i, zp=zeropad)
        if verbose:
            print("Generating {0}".format(img_number))

        # Determine image size to crop.
        rawlen = random_sampler()
        xc = np.random.randint(0, img.size[0] - rawlen)
        yc = np.random.randint(0, img.size[1] - rawlen)
        box = np.array([xc, yc, xc + rawlen, yc + rawlen], dtype='int32')

        # Load necessary because crop may be a lazy operation; im.load() should
        # copy it.  See <http://pillow.readthedocs.io/en/3.1.x/
        # reference/Image.html>.
        im = img.crop(box)
        im.load()

        # Obtain long/lat bounds for coordinate transform.
        ix = box[::2]
        iy = box[1::2]
        llong, llat = pix2coord(ix, iy, cdim, list(img.size), origin=origin)
        llbd = np.r_[llong, llat[::-1]]

        # Downsample image.
        im = im.resize([ilen, ilen], resample=Image.NEAREST)

        # Remove all craters that are too small to be seen in image.
        ctr_sub = ResampleCraters(craters, llbd, im.size[1], arad=arad,
                                  minpix=minpix)

        # Convert Plate Carree to Orthographic.
        [imgo, ctr_xy, distortion_coefficient] = PlateCarree_to_Orthographic(
            im, None, llbd, ctr_sub, iglobe=iglobe, ctr_sub=True, arad=arad,
            origin=origin, rgcoeff=1.2, slivercut=0.5)

        if imgo is None:
            print("Discarding narrow image")
            continue

        imgo_arr = np.asanyarray(imgo)
        assert imgo_arr.sum() > 0, ("Sum of imgo is zero!  There likely was "
                                    "an error in projecting the cropped "
                                    "image.")

        # Make target mask.  Used Image.BILINEAR resampling because
        # Image.NEAREST creates artifacts.  Try Image.LANZCOS if BILINEAR still
        # leaves artifacts).
        tgt = np.asanyarray(imgo.resize((tglen, tglen),
                                        resample=Image.BILINEAR))
        mask = make_mask(ctr_xy, tgt, binary=binary, rings=rings,
                         ringwidth=ringwidth, truncate=truncate)

        # Output everything to file.
        imgs_h5_inputs[i, ...] = imgo_arr
        imgs_h5_tgts[i, ...] = mask

        sds_box = imgs_h5_box.create_dataset(img_number, (4,), dtype='int32')
        sds_box[...] = box
        sds_llbd = imgs_h5_llbd.create_dataset(img_number, (4,), dtype='float')
        sds_llbd[...] = llbd
        sds_dc = imgs_h5_dc.create_dataset(img_number, (1,), dtype='float')
        sds_dc[...] = np.array([distortion_coefficient])

        craters_h5[img_number] = ctr_xy

        imgs_h5.flush()
        craters_h5.flush()
        i += 1

    imgs_h5.close()
    craters_h5.close()


def GenCroppedDataset(img, craters, outhead, rawlen_range=[1000, 2000],
                      rawlen_dist='log', rawlen_coeff=[2., 1.5],
                      ilen=256, cdim=[-180., 180., -60., 60.],
                      arad=1737.4, minpix=0, tglen=256, ringwidth=1,
                      truncate=True, amt=100, istart=0, seed=None,
                      verbose=False):
    """Generates random dataset from a global DEM and crater catalogue.

    The function randomly samples images from a global digital elevation
    map (DEM) that uses a Plate Carree projection.  The size of each image is
    (width, height) = (rawlen * x_coeff, rawlen * y_coeff), where rawlen is
    selected from a user specified distribution.  The image is then
    downsampled such that its central (rawlen, rawlen) region now has size
    (ilen, ilen) - we generally use ilen = 256.  The downsampled image is then
    converted from Plate Carree to orthographic projection.  Finally all pixels
    beyond the central (rawlen, rawlen) region are discarded.  This method of
    sampling and transforming a larger image, and then cropping its central
    region to obtain a final one, results in images that are relatively
    distortion-free and don't have black borders at their edges.

    Pixel coordinates and radii of catalogue craters that fall within each
    image's long/lat bounds are used to generate a corresponding binary ring
    mask.

    Input and target images are saved to disk in hdf5 format.  Crater
    positions and radii are also saved as a Pandas hdf5 store.

    Parameters
    ----------
    img : PIL.Image.Image
        Source image.
    craters : pandas.DataFrame
        Crater catalogue .csv.
    outhead : str
        Filepath and file prefix of the image and crater table hdf5 files.
    rawlen_range : list-like, optional
        Lower and upper bounds, in pixels, of the raw image width distribution.
        When sampling, a rawlen is randomly chosen from within this range,
        and a region (rawlen * x_coeff, rawlen * y_coeff) is sampled from the
        source image.  To always sample the same sized image, set lower bound
        to the same value as the upper.  Default is [1000, 2000].
    rawlen_dist : 'uniform' or 'log', optional
        Shape of the raw image width distribution.  'uniform' is uniform
        sampling, and 'log' is loguniform sampling.
    rawlen_coeff : list, optional
        [x_coeff, y_coeff] multipliers used to sample from source.  Default is
        [2., 1.5].
    ilen : int, optional
        Input image width, in pixels.  Sampled images will be downsampled
        such that (rawlen, rawlen) is mapped to (ilen, ilen).  Default is
        256. cdim : list-like, optional
        Coordinate limits (x_min, x_max, y_min, y_max) of image.  Default is
        LRO-Kaguya's [-180., 180., -60., 60.].
    arad : float. optional
        World radius in km.  Defaults to Moon radius (1737.4 km).
    minpix : int, optional
        Minimum crater diameter in pixels to be included in crater list.
        Useful when the smallest craters in the catalogue are smaller than 1
        pixel in diameter.
    tglen : int, optional
        Target image width, in pixels.
    ringwidth : int, optional
        Width (dr) of rings in the binary ring mask.
    truncate : bool
        Truncate mask wherever there is padding in the image.
    amt : int, optional
        Number of images to produce.  100 by default.
    istart : int
        Output file starting number, when creating datasets spanning multiple
        files.
    seed : int or None
        np.random.seed input (for testing purposes).
    verbose : bool
        If True, prints out number of image being generated.
    """

    # just in case we ever make this user-selectable...
    origin = "upper"

    # Seed random number generator.
    np.random.seed(seed)

    # Get craters.
    AddPlateCarree_XY(craters, list(img.size), cdim=cdim, origin=origin)

    iglobe = ccrs.Globe(semimajor_axis=arad*1000., semiminor_axis=arad*1000.,
                        ellipse=None)

    # Create random sampler (either uniform or loguniform).
    if rawlen_dist == 'log':
        rawlen_min = np.log10(rawlen_range[0])
        rawlen_max = np.log10(rawlen_range[1])

        def random_sampler():
            return int(10**np.random.uniform(rawlen_min, rawlen_max))
    else:

        def random_sampler():
            return np.random.randint(rawlen_range[0], rawlen_range[1] + 1)

    # Initialize output hdf5s.
    imgs_h5 = h5py.File(outhead + '_images.hdf5', 'w')
    imgs_h5_inputs = imgs_h5.create_dataset("input_images", (amt, ilen, ilen),
                                            dtype='uint8')
    imgs_h5_inputs.attrs['definition'] = "Input image dataset."
    imgs_h5_tgts = imgs_h5.create_dataset("target_masks", (amt, tglen, tglen),
                                          dtype='float32')
    imgs_h5_tgts.attrs['definition'] = "Target mask dataset."
    imgs_h5_llbd = imgs_h5.create_group("longlat_bounds")
    imgs_h5_llbd.attrs['definition'] = ("(long min, long max, lat min, "
                                        "lat max) of the PRE-CROP image.")
    imgs_h5_pcsize = imgs_h5.create_group("precrop_size")
    imgs_h5_pcsize.attrs['definition'] = ("(width, height) of PRE-CROP image, "
                                          "in pixels.")
    imgs_h5_cll = imgs_h5.create_group("cll_xy")
    imgs_h5_cll.attrs['definition'] = ("Pixel coordinates of central long /"
                                       " lat within CROP image.")
    imgs_h5_ppkm = imgs_h5.create_group("pixperkm")
    imgs_h5_ppkm.attrs['definition'] = ("Conversion factor from km to image "
                                        "pixel.")
    craters_h5 = pd.HDFStore(outhead + '_craters.hdf5', 'w')

    # Zero-padding for hdf5 keys.
    zeropad = int(np.log10(amt)) + 1

    for i in range(amt):

        # Current image number.
        img_number = "img_{i:0{zp}d}".format(i=istart + i, zp=zeropad)
        if verbose:
            print("Generating {0}".format(img_number))

        # Determine image size to crop.
        rawlen = random_sampler()
        rawlen_withpad = [np.ceil(rawlen_coeff[0] * rawlen).astype(int),
                          np.ceil(rawlen_coeff[1] * rawlen).astype(int)]
        xc = np.random.randint(0, img.size[0] - rawlen_withpad[0])
        yc = np.random.randint(0, img.size[1] - rawlen_withpad[1])
        box = np.array([xc, yc, xc + rawlen_withpad[0],
                        yc + rawlen_withpad[1]], dtype='int32')

        # Load necessary because crop may be a lazy operation; im.load() should
        # copy it.  See <http://pillow.readthedocs.io/en/3.1.x/
        # reference/Image.html>.
        im = img.crop(box)
        im.load()

        # Obtain long/lat bounds for coordinate transform.
        ix = box[::2]
        iy = box[1::2]
        llong, llat = pix2coord(ix, iy, cdim, list(img.size), origin=origin)
        llbd = np.r_[llong, llat[::-1]]

        # Downsample image.
        im = im.resize([np.ceil(rawlen_coeff[0] * ilen).astype(int),
                        np.ceil(rawlen_coeff[1] * ilen).astype(int)],
                       resample=Image.NEAREST)

        # Remove all craters that are too small to be seen in image.
        ctr_sub = ResampleCraters(craters, llbd, im.size[1], arad=arad,
                                  minpix=minpix)

        # Convert Plate Carree to Orthographic.
        [imgo, ctr_xy, pixperkm, imgo_uc_size, clonglat_xy] = (
            PlateCarree_to_Orthographic_Cropped(
                im, ilen, llbd, ctr_sub, iglobe=iglobe, arad=arad,
                origin=origin, rgcoeff=1.3, paderr=0.5))

        imgo_arr = np.asanyarray(imgo)
        assert imgo_arr.sum() > 0, ("Sum of imgo is zero!  There likely was "
                                    "an error in projecting the cropped "
                                    "image.")

        # Make target mask.  Used Image.BILINEAR resampling because
        # Image.NEAREST creates artifacts.  Try Image.LANZCOS if BILINEAR still
        # leaves artifacts).
        tgt = np.asanyarray(imgo.resize((tglen, tglen),
                                        resample=Image.BILINEAR))
        # Generate binary ring mask.
        mask = make_mask(ctr_xy, tgt, binary=True, rings=True,
                         ringwidth=ringwidth, truncate=truncate)

        # Output everything to file.
        imgs_h5_inputs[i, ...] = imgo_arr
        imgs_h5_tgts[i, ...] = mask

        sds_llbd = imgs_h5_llbd.create_dataset(img_number, (4,), dtype='float')
        sds_llbd[...] = llbd
        sds_pcsize = imgs_h5_pcsize.create_dataset(img_number, (2,),
                                                   dtype='int')
        sds_pcsize[...] = np.array(imgo_uc_size)
        sds_cll = imgs_h5_cll.create_dataset(img_number, (2,), dtype='float')
        sds_cll[...] = clonglat_xy.loc[:, ['x', 'y']].as_matrix().ravel()
        sds_ppkm = imgs_h5_ppkm.create_dataset(img_number, (1,), dtype='float')
        sds_ppkm[...] = np.array([pixperkm])

        craters_h5[img_number] = ctr_xy

        imgs_h5.flush()
        craters_h5.flush()
        i += 1

    imgs_h5.close()
    craters_h5.close()
