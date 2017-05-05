"""
Charles's attempt at creating a convnet that translates an image
into a density map, which can then be postprocessesed to give a
crater count.
"""

# Past-proofing
from __future__ import absolute_import, division, print_function

def vgg16(n_classes,im_width,im_height,learn_rate,lmbda,dropout):
    n_filters = 32          #vgg16 uses 64
    n_blocks = 3            #vgg16 uses 5
    n_dense = 512           #vgg16 uses 4096

    #first block
    print('Making VGG model...')
    model = Sequential()
    model.add(Conv2D(n_filters, nb_row=3, nb_col=3, activation='relu', border_mode='same', W_regularizer=l2(lmbda), input_shape=(im_width,im_height,3)))
    model.add(Conv2D(n_filters, nb_row=3, nb_col=3, activation='relu', border_mode='same', W_regularizer=l2(lmbda)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #subsequent blocks
    for i in np.arange(1,n_blocks):
        n_filters_ = np.min((n_filters*2**i, 512))                          #maximum of 512 filters in vgg16
        model.add(Conv2D(n_filters_, nb_row=3, nb_col=3, activation='relu', border_mode='same', W_regularizer=l2(lmbda)))
        model.add(Conv2D(n_filters_, nb_row=3, nb_col=3, activation='relu', border_mode='same', W_regularizer=l2(lmbda)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(n_dense, activation='relu', W_regularizer=l2(lmbda)))   #biggest memory sink
    model.add(Dropout(dropout))
    model.add(Dense(n_dense, activation='relu', W_regularizer=l2(lmbda)))
    model.add(Dense(n_classes, activation='relu', name='predictions'))      #relu/regression output

    #optimizer = SGD(lr=learn_rate, momentum=0.9, decay=0.0, nesterov=True)
    optimizer = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
    print model.summary()
    return model
