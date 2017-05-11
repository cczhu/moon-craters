"""
Charles's attempt at creating an I/O pipeline and convnet for crater counting.
"""

################ IMPORTS ################

# Past-proofing
from __future__ import absolute_import, division, print_function

# System modules
import os
import sys
import glob
#import cv2
import datetime

# I/O and math stuff
import pandas as pd
import numpy as np
from PIL import Image

sys.path.append("/home/m/mhvk/czhu/moon_craters")
import make_density_map as densmap

# NN and CV stuff
from sklearn.model_selection import KFold, train_test_split
from keras import backend as K
K.set_image_dim_ordering('tf')
import keras
#from keras.preprocessing.image import ImageDataGenerator
import keras.preprocessing.image as kpimg

################ DATA READ-IN FUNCTIONS (FROM moon4.py and moon_vgg16_1.2.2.py) ################

def get_im_csv(path, args):
    """Grabs image (greyscale) using PIL.Image, converts it to
    np.array, then grabs craters as pd.DataFrame
    """
    img = Image.open(path).convert('L')
    img = np.asanyarray(img.resize(args["imgshp"]))
    craters = pd.read_csv(path.split(".png")[0] + ".csv")
    craters.drop( np.where(craters["Diameter (pix)"] < args["c_pix_cut"])[0], 
                        inplace=True )
    craters.reset_index(drop=True, inplace=True)
    return img, craters


def load_data(path, args):
    """Chain-loads data.
    """
    X = []
    X_id = []
    ctrs = []
    files = glob.glob('%s*.png'%path)
    print("number of files: %d"%(len(files)))
    for fl in files:
        flbase = os.path.basename(fl)
        img, craters  = get_im_csv(fl, args)
        X.append(img)
        X_id.append(fl)
        ctrs.append(craters)
    return X, ctrs, X_id


def read_and_normalize_data(path, args, data_flag):
    """Reads and normalizes input data.  Removes craters below some
    minimum size.
    """
    print("For {0:s} data".format(data_flag))
    X, ctrs, X_id = load_data(path, args)
    # Convert to np.array and normalize
    X = np.array(X, dtype=np.float32) / 255.
    print('Shape:', X.shape)
    return X, ctrs, X_id


################ TRAINING ROUTINE ################

def run_cross_validation_process_test(info_string, model):
	batch_size = 32
	num_fold = 0
	yfull_test = []
	test_id = []

	test_data,test_target, test_id = read_and_normalize_test_data()
	test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)

	result1 = pd.Series(test_id, test_prediction)
	now = datetime.datetime.now()
	sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
	result1.to_csv(sub_file, index=False)	

	#	score = mean_absolute_error(test_target, test_prediction)
	#	print('Holdout set score is: ', score)


def run_training(train_data, train_ctrs, test_data, test_ctrs, args):
    


def run_cross_validation(train_data, train_ctrs, test_data, test_ctrs, args):

    #Squash train_target (e.g. from 0-10 -> 0-1 crater counts)
    #train_target = np.log10(1+train_target)

    #Iterate
    N_runs = 6
    lmbda = random.sample(np.logspace(-3,1,5*N_runs), N_runs-1)
    dropout = random.sample(np.linspace(0,0.8,5*N_runs), N_runs-1)
    lmbda.append(0), dropout.append(0)  #ensure we have a baseline comparison
    for i in range(N_runs):
        l,d = lmbda[i], dropout[i]
        score = train_test_model(train_data,train_target,test_data,test_target,learn_rate,batch_size,l,d,nb_epoch,n_train_samples,im_width,im_height,n_classes,rs)
        print '###################################'
        print '##########END_OF_RUN_INFO##########'
        print('\nTest Score is %f.\n'%score)
        print 'learning_rate=%e, batch_size=%d, lambda=%e, dropout=%f, n_epoch=%d, n_train_samples=%d, n_classes=%d, random_state=%d, im_width=%d, im_height=%d'%(learn_rate,batch_size,l,d,nb_epoch,n_train_samples,n_classes,rs,im_width,im_height)
        print '###################################'
        print '###################################'


################ MAIN ################

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Keras-based CNN for mapping crater images to density maps.')
    parser.add_argument('--path', type=str, required=False,
                        help='Filepath of lola pngs and csvs', default="./")
    parser.add_argument('--learn_rate', type=float, required=False,
                        help='Learning rate', default=0.0001)
    parser.add_argument('--imshape', nargs=2, type=int, required=False,
                        help='[height, length] of input image to convnet')
    parser.add_argument('--crater_cutoff', type=int, required=False,
                        help='Crater pixel diameter cutoff', default=3)
    parser.add_argument('--batchsize', type=int, required=False,
                        help='Crater pixel diameter cutoff', default=32)
    parser.add_argument('--lambd', type=float, required=False,
                        help='L2 regularization coefficient', default=0.)
    parser.add_argument('--epochs', type=int, required=False,
                        help='Number of training epochs', default=30)
    parser.add_argument('--f_samp', type=int, required=False,
                        help='Random fraction of samples to use', default=1.)


#    parser.add_argument('--lu_csv_path', metavar='lupath', type=str, required=False,
#                        help='Path to LU78287 crater csv.', default="./LU78287GT.csv")
#    parser.add_argument('--alan_csv_path', metavar='lupath', type=str, required=False,
#                        help='Path to LROC crater csv.', default="./alanalldata.csv")
#    parser.add_argument('--outhead', metavar='outhead', type=str, required=False,
#                        help='Filepath and filename prefix of outputs.', default="out/lola")
#    parser.add_argument('--amt', type=int, default=7500, required=False,
#                        help='Number of images each thread will make (multiply by number of \
#                        threads for total number of images produced).')
    in_args = parser.parse_args()

    # Declare master dictionary of input variables
    args = {}

    # Load constants from user
    args["path"] = args.path
    args["learn_rate"] = in_args.learning_rate
    args["c_pix_cut"] = in_args.crater_cutoff

    args["imgshp"] = (300, 300)
    if in_args.imshape:
        args["imgshp"] = in_args.imshape

    args["batchsize"] = in_args.batchsize
    args["lambda"] = in_args.lambd
    args["epochs"] = in_args.epochs
    args["f_samp"] = in_args.f_samp

    # Print Keras version, just in case
    print('Keras version: {0}'.format(keras.__version__))

    # Try to load data from working directory
    # The .npy stuff doesn't work for pandas dataframes
#    try:
#        train_data = np.load(args["path"] + '/training_set/train_data.npy')
#        train_ctrs = np.load(args["path"] + '/training_set/train_ctrs.npy')
#        test_data = np.load(args["path"] + '/test_set/test_data.npy')
#        test_ctrs = np.load(args["path"] + '/test_set/test_ctrs.npy')
#        print("Successfully loaded .npy files from working directory.")
#    except:
    print("Can't find .npy files locally; reading in from args.path.")
    train_data, train_ctrs, train_id = \
                read_and_normalize_data(args["path"] + "/training_set/", 
                                        args, "train")
    test_data, test_ctrs, test_id = \
                read_and_normalize_data(args["path"] + "/test_set/", 
                                        args, "test")
#        np.save(args["path"] + '/training_set/train_data.npy', train_data)
#        np.save(args["path"] + '/training_set/train_ctrs.npy', train_ctrs)
#        np.save(args["path"] + '/test_set/test_data.npy', test_data)
#        np.save(args["path"] + '/test_set/test_ctrs.npy', test_ctrs)

    # Calculate next largest multiple of batchsize to N_train*f_samp
    # Then use to obtain subset
    N_sub = int(args["batchsize"] * np.ceil( train_data.shape[0] * \
                                        args["f_samp"] / args["batchsize"] ))
    subset = np.random.choice(train_data.shape[0], size=N_sub)
    train_data = train_data[subset]
    train_target = train_target[subset]


























    gen = ImageDataGenerator(#channel_shift_range=30,                    #R,G,B shifts
                             #rotation_range=180,                        #rotations
                             width_shift_range=1./im_width,
                             height_shift_range=1./im_height,
                             fill_mode='constant',
                             horizontal_flip=True,vertical_flip=True)    #flips

class MoonImageGen(object)
    """Heavily modified version of keras.preprocessing.image.ImageDataGenerator.
    that creates density maps or masks, and performs random transformation
    on both source image and result consistently by treating the map/mask
    as another colour channel.

    Parameters
    ----------
    maptype : str
        "dens" or "mask"
    samplewise_center: bool
        Set each sample mean to 0.
    samplewise_std_normalization: bool
        Divide each input by its std.
    rotation_range: float (0 to 180)
        Range of possible rotations (from -rotation_range to 
        +rotation_range).
    width_shift_range: float (0 to 1)
        +/- fraction of total width that can be shifted.
    height_shift_range: float (0 to 1)
        +/- fraction of total height that can be shifted.
    shear_range: float
        Shear intensity (shear angle in radians).
    zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
        in the range [1-z, 1+z]. A sequence of two can be passed instead
        to select this range.
    fill_mode: points outside the boundaries are filled according to the
        given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
        is 'constant', which works with default cval=0 to keep regions
        outside border zeroed.
    cval: value used for points outside the boundaries when fill_mode is
        'constant'. Default is 0.
    horizontal_flip: whether to randomly flip images horizontally.
    vertical_flip: whether to randomly flip images vertically.
    rescale: pixel rescaling factor ("contrast"). If None or 0, no rescaling is applied,
        otherwise we multiply the data by the value provided
        (before applying any other transformation).
    data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
        (the depth) is at index 1, in 'channels_last' mode it is at index 3.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
    """
    def __init__(self,
                 maptype="density",
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 fill_mode='constant',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 data_format=None):

        self.maptype = maptype

        if data_format is None:
            data_format = K.image_data_format()
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('data_format should be "channels_last" (channel after row and '
                             'column) or "channels_first" (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('zoom_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)


    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpeg',
                            follow_links=False):
        assert Exception("flow_from_directory is currently not supported by MoonImageGen!")


    def fit(self, x, augment=False, rounds=1, seed=None):
        """Dummy function to prevent users from
        fitting entire dataset."""
        assert Exception("fit is currently not supported by MoonImageGen!")


    def flow(self, x, ctrs, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return MoonIterator(
            x, ctrs, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class MoonIterator(Iterator):
    """Iterator yielding data from input numpy array and crater csv.

    Parameters
    ----------
    x: numpy.array
        Numpy array of input data.
    ctrs: list
        List of crater csvs.
    moon_image_gen: MoonImageGen instance
        Instance to use for map creation, random transformations 
        and normalization.
    batch_size: int
        Size of a batch.
    shuffle: bool
        Toggle whether to shuffle the data between epochs.
    seed: int
        Random seed for data shuffling.
    data_format: str
        One of `channels_first`, `channels_last`.
    """

    def __init__(self, x, ctrs, moon_image_gen,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None):
        if len(x) != len(ctrs):
            raise ValueError('X (images tensor) and ctrs (crater csvs) '
                             'should have the same length. '
                             'Found: X.shape = %s, ctrs.shape = %s' %
                             (np.asarray(x).shape, len(y)))

        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(x, dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('NumpyArrayIterator is set to use the '
                             'data format convention "' + data_format + '" '
                             '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                             'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                             'However, it was passed an array with shape ' + str(self.x.shape) +
                             ' (' + str(self.x.shape[channels_axis]) + ' channels).')
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)


    def next(self):
        """Use next for python 2.x, __next__ for 3.x (class Iterator returns
        self.next(*args, **kwargs) )

        Returns
        -------
        batch_x: numpy.array
            Next batch of crater images
        batch_y: numpy.array
            Next batch of crater maps
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.image_data_generator.random_transform(x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y
