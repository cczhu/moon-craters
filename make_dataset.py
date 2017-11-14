#!/usr/bin/env python
"""Input Data Generator for DeepMoon Convnet Crater Detector

Script for generating input datasets from Lunar global digital elevation maps 
(DEMs) and crater catalogs.

We used the USGS Astrogeology Cloud Processing service to convert the
LRO-Kaguya merged DEM - found at https://astrogeology.usgs.gov/
search/map/Moon/LRO/LOLA/Lunar_LRO_LrocKaguya_DEMmerge_60N60S_512ppd - to
16-bit GeoTiff format, and reduce the resolution to 118 m/pixel.  (We also
tried converting to 8-bit GeoTiff, but Astrocloud, rather than rescaling,
simply maps values beyond the 8-bit range to either 0 or 255).  The GDAL
library was then used to convert this to an 8-bit png at the same resolution:

gdal_translate -of PNG -scale -21138 21138 -co worldfile=no 
    LunarLROLrocKaguya_118mperpix_int16.tif LunarLROLrocKaguya_118mperpix.png

This script is designed to use the LRO-Kaguya DEM and a combination of the
LOLA-LROC 5 - 20 km and Head et al. 2010 >=20 km crater catalogs.  It
generates a randomized set of small (projection-corrected) images and
corresponding crater targets.  The input and target image sets are stored as
hdf5 files.  The longitude and latitude limits of each image is included in the
input set file, and tables of the craters in each image are stored in a
separate Pandas HDFStore hdf5 file.

The script's parameters are located under the Global Variables.  We recommend
making a copy of this script when generating a dataset.

MPI4py can be used to generate more images - each thread is given `amt` number
of images to generate.  Comment out the MPI code block below to run on systems
where it's not installed.
"""

########## Imports ##########

# Past-proofing
from __future__ import absolute_import, division, print_function

# I/O and math stuff
import pandas as pd
import numpy as np
from PIL import Image

# Input making modules
import make_input_data as mkin
import make_density_map as densmap

########## Global Variables ##########

# Source image path.
source_image_path = "/home/cczhu/public_html/LOLA_Global_20k.png"
# Head et al. dataset csv path.
head_csv_path = "./LolaLargeCraters.csv"
# LROC crater dataset (from Alan) csv path.
alan_csv_path = "./alanalldata.csv"
# Output filepath and file header.  Eg. if outhead = "./input_data/train",
# files will have extension "./out/train_inputs.hdf5", "./out/train_inputs.hdf5"
outhead = "./input_data/train"

# Number of images to make (if using MPI4py, number of image per thread to
# make).
amt = 60000

# Range of image widths, in pixels, to crop from source image.  For Orthogonal
# projection, crater radii are changed roughly by a factor of
# cos(dphi * pi / 360), where dphi is the angular height of the image.  Larger
# images are distorted at their edges but 
ilen_range = [600., 2000.]

olen = 256                                          # Size of moon images
dmlen = 256                                         # Size of density maps (should be 2^i smaller than olen, for 
                                                    # some integer i >=0 dependent on CNN architecture)

source_cdim = [-180, 180, -90, 90]                  # [Min long, max long, min lat, max lat] dimensions of source 
                                                    # image (it'll almost certainly be the entire globe) DO NOT ALTER

sub_cdim = [-180, 180, -90, 90]                     # [Min long, max long, min lat, max lat] sub-range of long/lat to
                                                    # use when sampling random images, useful for separating train and 
                                                    # test sets

minpix = 1.                                         # Minimum pixel diameter of craters used in density map.  
                                                    # 5 km on the moon is 0.5 degrees, or ~60 pixels for a ~20k image.

slivercut = 0.8                                     # Minimum width/height aspect ratio of acceptable image.  Keeping this
                                                    # < 0.8 or so prevents "wedge" images derived from polar regions from
                                                    # being created.  DO NOT SET VALUE TOO CLOSE TO UNITY!

outp = "outp"                                       # If str, script dumps pickle containing the long/lat boundary and crop 
                                                    # bounds of all images.  Filename will be of the form outhead + outp + ".p".
                                                    # If multithreading is enabled, rank will be appended to filename.


# Physical arguments

R_km = 1737.4                                       # Radius of the world in km - 1737.4 for Moon, 2439.7 for Mercury


# Density map and mask arguments

maketype = "mask"                                   # Type of target to make - "dens" for density map, "mask" for mask

savetiff = True                                     # If true, save density maps as tiff files (8-bit pngs don't work for intensity maps)
                                                    # with arbitrary scaling.

savenpy = True                                      # If true, dumps input images to outhead + "input.npy" , and target density maps or masks to
                                                    # outhead + "targets.npy"

dmap_args = {}                                      # dmap kernel args

dmap_args["truncate"] = True                        # If True, truncate mask where image truncates (i.e. has padding rather than image content)


# Density map args
dmap_args["kernel"] = None                          # Specifies type of kernel to use.  Can be a function, "knn" (k nearest neighbours), 
                                                    # or None.  If a function is inputted, function must return an array of 
                                                    # length craters.shape[0].  If "knn",  uses variable kernel with 
                                                    #    sigma = beta*<d_knn>,
                                                    # where <d_knn> is the mean Euclidean distance of the k = knn nearest 
                                                    # neighbouring craters.  If anything else is inputted, will use
                                                    # constant kernel size with sigma = k_sigma.

dmap_args["k_support"] = 8                          # Kernel support (i.e. size of kernel stencil) coefficient.  Support
                                                    # is determined by kernel_support = k_support*sigma.

dmap_args["k_sig"] = 3.                             # Sigma for constant sigma kernel (kernel = None).

dmap_args["knn"] = 10                               # k nearest neighbours, used when kernel = "knn".

dmap_args["beta"] = 0.2                             # Beta value used to calculate sigma when kernel = "knn" (see above).

dmap_args["kdict"] = {}                             # If kernel is custom function, dictionary of arguments passed to kernel.


# Mask arguments

dmap_args["rings"] = False                          # If True, use rings as masks rather than circles

dmap_args["ringwidth"] = 1                          # If dmap_args["rings"] = True, thickness of ring

dmap_args["binary"] = True                          # If True, returns a binary image of crater masks 


# Determine outp, and set rank = 0 in case MPI is not used below
outp = outp + ".p"
rank = 0


########################### Functions ###########################


def load_img_make_target(filename, maketype, outshp, minpix, dmap_args):
    """Loads individual image.
    """
    # Load base image
    img = Image.open(filename).convert('L')
    # Dummy image of target size.  Bilinear interpolation is compromise
    # between Image.NEAREST, which creates artifacts, and Image.LANZCOS,
    # which is more expensive (though try that one if BILINEAR gives
    # crap)
    omg = np.asanyarray(img.resize(outshp, resample=Image.BILINEAR))
    img = np.asanyarray(img)

    # Load craters CSV
    craters = pd.read_csv(filename.split(".png")[0] + ".csv")
    # Resize crater positions and diameters to target size
    craters.loc[:, ["x", "y", "Diameter (pix)"]] *= outshp[0]/img.shape[0]
    # Cut craters that are now too small to be detected clearly
    craters = craters[craters["Diameter (pix)"] >= minpix]
    craters.reset_index(inplace=True, drop=True)

    if maketype == "mask":
        dmap = densmap.make_mask(craters, omg, binary=dmap_args["binary"],
                                        rings=dmap_args["rings"],
                                        ringwidth=dmap_args["ringwidth"],
                                        truncate=dmap_args["truncate"])
    else:
        dmap = densmap.make_density_map(craters, omg, kernel=dmap_args["kernel"], 
                        k_support=dmap_args["k_support"], 
                        k_sig=dmap_args["k_sig"], knn=dmap_args["knn"], 
                        beta=dmap_args["beta"], kdict=dmap_args["kdict"], 
                        truncate=dmap_args["truncate"])

    return img, dmap


def make_dmaps(files, maketype, outshp, minpix, dmap_args, savetiff=False):
    """Chain-loads input data pngs and make target density maps/masks

    Parameters
    ----------
    files : list
        List of files to process
    maketype : str
        "dens" or "mask", depending on if you want to make
        a density map or a mask
    outshp : listlike
        [height, width] of target image
    minpix : float
        Minimum crater diameter in pixels to be included in target
    dmap_args : dict
        Dictionary of arguments to pass to target generation 
        functions.
    savetiff : bool
        If True, saves target to output file with name = 
        filename.split(".png") + maketype + ".tiff".  Using
        tiff as file format because target is density map with
        arbitrary intensities, while most image formats go from 
        0 - 256 between 3 channels.        
    """
    cX0, cY0 = load_img_make_target(files[0], maketype, outshp, minpix,
        dmap_args)

    X = np.empty((len(files),) + cX0.shape, dtype=np.uint8)
    Y = np.empty((len(files),) + cY0.shape, dtype=np.float32)

    #files = sorted([fn for fn in glob.glob('%s*.png'%path)
    #         if (not os.path.basename(fn).endswith('mask.png') and
    #        not os.path.basename(fn).endswith('dens.png'))])
    print("Number of input images generated: %d"%(len(files)))
    print("Generating target images ({0:s}).".format(maketype))

    for i, fl in enumerate(files):
        cX, cY = load_img_make_target(fl, maketype, outshp, minpix, dmap_args)
        X[i] = cX
        Y[i] = cY
        mname = fl.split(".png")[0] + maketype + ".tiff"
        if savetiff:
            imgo = Image.fromarray(cY)
            imgo.save(mname);

    return X, Y


########################### Script ###########################


if __name__ == '__main__':

    ########################### MPI ###########################

    # Utilize mpi4py for multithreaded processing
    # Comment this block out if mpi4py is not available
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print("Thread {0} of {1}".format(rank, size))
    if outp: # Append rank to outp filename
        outp = outp.split(".p")[0] + "_p{0}.p".format(rank)

    ########################### MPI ###########################

    # Read source image
    img = Image.open(source_image_path).convert("L")
        
    if sub_cdim != source_cdim:
        img = mkin.InitialImageCut(img, source_cdim, sub_cdim)

    craters = mkin.ReadLROCHeadCombinedCraterCSV(filelroc=alan_csv_path,
                                                 filehead=head_csv_path)
    # Co-opt ResampleCraters to remove all craters beyond subset cdim
    # keep minpix = 0 (since we don't have pixel diameters yet)
    craters = mkin.ResampleCraters(craters, sub_cdim, None, arad=R_km)

    # Generate input images
    print("Generating input images")
    mkin.GenDataset(img, craters, outhead, ilen_range=ilen_range, olen=olen,
                    cdim=sub_cdim, arad=R_km, amt=amt, zeropad=zeropad,
                    minpix=minpix, slivercut=slivercut, outp=outp,
                    istart = rank*amt)

    files = [outhead + "_{i:0{zp}d}.png".format(i=i, zp=zeropad) 
                                        for i in range(rank*amt, (rank+1)*amt)]

    # Generate target density maps/masks
    outshp = (dmlen, dmlen)
    X, Y = make_dmaps(files, maketype, outshp, minpix, dmap_args, 
                      savetiff=savetiff)

    # Optionally, save data as npy file
    if savenpy:
        np.save(outhead + "_{rank:01d}_input.npy".format(rank=rank), X)
        np.save(outhead + "_{rank:01d}_targets.npy".format(rank=rank), Y)
