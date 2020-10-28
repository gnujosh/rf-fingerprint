# -----------------------------------------------------------------------------------
# ---------------------------------- UNCLASSIFIED -----------------------------------
# This code was developed under the Defense Advanced Research Projects Agency (DARPA)
# Radio Frequency Machine Learning System (RFMLS) program: contract FA8750-18-C-0150.
# Government Unlimited Rights
# BAE Systems, Inc. (C) 2020.
# -----------------------------------------------------------------------------------

"""MarconiNet models constructed from marconinet_blocks.py components."""

# third party imports
import tensorflow.keras as keras

# local application imports
from .marconinet_blocks import sigDCCLayer
from .marconinet_blocks import sigDCCResBlockStack
from .marconinet_blocks import sigConv_and_pooling_block


def get_marconinet_reconstruction_model(
    input_shape,
    rb_dilation_rate=[1, 4, 16, 64, 192, 256],
    rb_filters=32,
    rb_kernel_size=[4, 4, 4, 4, 4, 4],
    post_activation="relu",
):
    """Returns a MarconiNet Reconstruction Model (MRM) where decoder is only a linear
    transform.

    # Arguments

        input_shape: (tuple of integers) model input shape minus the batch dimension.
        rb_dilation_rate: (list of integers) dilation rates for the residual block stack.
        rb_kernel_size: (list of integers) kernel sizes for the residual block stack.
        rb_filters: (integer) filters per block in the sigDCC stack.
        post_activation: (string) activation function to use for each residual block.

    # Returns

        keras.models.Model, the reconstruction model
    """
    # build encoder
    input = keras.layers.Input(shape=input_shape)
    marconi_encoding = sigDCCResBlockStack(
        input,
        rb_dilation_rate=rb_dilation_rate,
        rb_filters=rb_filters,
        rb_kernel_size=rb_kernel_size,
        post_activation=post_activation,
        index="mrm",
    )
    encoder = keras.models.Model(inputs=input, outputs=marconi_encoding, name="encoder")
    # build autoencoder
    linear = keras.layers.Dense(1, name="reconstruction_layer")(marconi_encoding)
    autoencoder = keras.models.Model(
        inputs=input, outputs=linear, name="marconinet_reconstruction_model"
    )
    return autoencoder


def get_marconinet_classifier(
    input_shape,
    n_classes,
    rb_dilation_rate=[2, 4, 8, 16, 32, 64, 128, 256],
    rb_filters=50,
    rb_kernel_size=8 * [4],
    post_activation="relu",
    post_cblocks=3,
    cblock_depth=3 * [2],
    cblock_filters=[50, 50, 50],
    cblock_kernsz=3 * [4],
    cblock_poolrates=3 * [4],
    cblock_rescon=3 * [True],
):
    """Returns a sigDCC classifier model.

    # Arguments

        input_shape: (tuple of integers) model input shape minus the batch dimension.
        rb_dilation_rate: (list of integers) dilation rates for the residual block stack.
        rb_kernel_size: (list of integers) kernel sizes for the residual block stack.
        rb_filters: (integer) filters per block in the sigDCC stack.
        post_activation: (string) activation function to use for each residual block.
        post_cblocks: (integer) number of convolution + pooling blocks.
        cblock_depth: (list of integers) number of convolutions per cblock.
        cblock_filters: (list of integers) number of filters for cblock convolutions.
        cblock_kernsz: (list of integers) kernel sizes for cblock convolutions.
        cblock_poolrates: (list of integers) pooling rates for cblocks.
        cblock_rescon: (list of booleans) whether to use residual connections in cblocks.

    # Returns

        keras.models.Model, the reconstruction model
    """
    input = keras.layers.Input(shape=input_shape)
    # setup marconnet layers
    x = sigDCCLayer(input, filters=rb_filters, kernel_size=2, name="conv1")
    x = sigDCCResBlockStack(
        x,
        rb_dilation_rate=rb_dilation_rate,
        rb_filters=rb_filters,
        rb_kernel_size=rb_kernel_size,
        post_activation=post_activation,
        index="marconinet",
    )
    # add non-dilated convolution and pooling blocks (i.e. cblocks)
    for i in range(post_cblocks):
        blk_name = f"blk{i}"
        x = sigConv_and_pooling_block(
            x,
            cblock_filters[i],
            cblock_kernsz[i],
            conv_depth=cblock_depth[i],
            name=blk_name,
            res_con=cblock_rescon[i],
            pooling_rate=cblock_poolrates[i],
        )
    # add classification layers
    x = keras.layers.Flatten(name="cblock_flatten")(x)
    x = keras.layers.Dense(
        n_classes, activation="softmax", name="softmax_classification"
    )(x)
    model = keras.models.Model(inputs=input, outputs=x, name="marconinet_classifier")
    return model


if __name__ == "__main__":
    # don't use gpus for testing
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ","
    # try instantiating model and printing summary
    input_shape = (1600,2,1) # (sig_length, real/imag, dummy_dim)
    n_classes = 100
#    model = get_marconinet_reconstruction_model(input_shape)
    model = get_marconinet_classifier(input_shape, n_classes)
    model.summary()
