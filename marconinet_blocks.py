# -----------------------------------------------------------------------------------
# ---------------------------------- UNCLASSIFIED -----------------------------------
# This code was developed under the Defense Advanced Research Projects Agency (DARPA)
# Radio Frequency Machine Learning System (RFMLS) program: contract FA8750-18-C-0150.
# Government Unlimited Rights
# BAE Systems, Inc. (C) 2020.
# -----------------------------------------------------------------------------------

import tensorflow.keras as keras
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Lambda
import numpy as np


def sigDCCLayer(
    x,
    filters,
    kernel_size,
    dilation_rate=1,
    name=None,
    activation=None,
    padding="causal",
):
    """A 2D causal convolution layer that treats the rows as the time dimension with 2 
    columns for complex input--one column for the real part and one for the complex part.

    # Arguments

        x: (keras tensor) input tensor
        filters: (integer) number of filters for the layer.
        kernel_size: (integer) filter kernel size.
        dilation_rate: (integer) kernel dilation rate.
        name: (string) used for naming the layer
        activation: (string) activation function to use.
        padding: (string) "causal" or "same"

    # Returns

        The feature maps of the sigDCC convolution layer 
        with shape (batch, signal length, 2, filters).
    """

    # define layers
    padname = name if name == None else f"{name}_cpad"
    cr_name = name if name == None else f"{name}_cR"
    ci_name = name if name == None else f"{name}_cI"
    cat_name = name if name == None else f"{name}_cat"
    rsh_name = name if name == None else f"{name}_rsh"
    if padding == "causal":
        causal_pad = ZeroPadding2D(
            ((dilation_rate * (kernel_size - 1), 0), (0, 0)), name=padname
        )
    else:
        assert dilation_rate == 1, "if padding is 'same' then dilation_rate must be 1"
        if kernel_size % 2 == 0:
            pad_l = int(kernel_size / 2)
            pad_r = int(kernel_size / 2 - 1)
        else:
            pad_l = int(kernel_size // 2)
            pad_r = int(pad_l)
        causal_pad = ZeroPadding2D(((pad_l, pad_r), (0, 0)), name=padname)
    if activation is None:
        conv_real = Conv2D(
            filters,
            (kernel_size, 2),
            padding="valid",
            dilation_rate=(dilation_rate, 1),
            name=cr_name,
        )
        conv_imag = Conv2D(
            filters,
            (kernel_size, 2),
            padding="valid",
            dilation_rate=(dilation_rate, 1),
            name=ci_name,
        )
    else:
        conv_real = Conv2D(
            filters,
            (kernel_size, 2),
            padding="valid",
            dilation_rate=(dilation_rate, 1),
            activation=activation,
            name=cr_name,
        )
        conv_imag = Conv2D(
            filters,
            (kernel_size, 2),
            padding="valid",
            dilation_rate=(dilation_rate, 1),
            activation=activation,
            name=ci_name,
        )

    # define feed forward operations
    sig_length = x.get_shape().as_list()[1]
    x = causal_pad(x)
    xr = conv_real(x)
    xi = conv_imag(x)
    x = Concatenate(name=cat_name)([xr, xi])
    x = Reshape((sig_length, 2, filters), name=rsh_name)(x)
    return x


def sigDCCResidualBlock(
    x,
    j,
    rec_field,
    cov_factor,
    kernel_size=4,
    dilation_rate=2,
    n_filters=100,
    index=None,
):
    """Takes a tensor and a layer index and returns a gated residual block.

    This is the same residual block as defined in the original wavenet
    paper arXiv:1609.03499v2

    # Arguments

        x: (keras tensor) input tensor to the residual block.
        j: (integer) layer index for naming residual block layers.
        rec_field: (integer) receptive field of the input tensor x.
        cov_factor: (float) coverage factor of the input tensor x.
        kernel_size: (integer) specifies the kernel size of the residual block.
        dilation_rate: dilation factor for each residual block
        n_filters: (integer) to specify the number of filters of the residual block
        index: (string) for naming layers.

    # Returns

        res: the output tensor of the residual block
        skip: skip connection tensor of the residual block
    """
    # creat layer names
    cf_percent = int(np.floor(cov_factor * 100.0))
    tanh_name = index if index == None else f"{index}_tanh-{rec_field}-{cf_percent}"
    sig_name = index if index == None else f"{index}_sig-{rec_field}-{cf_percent}"
    mult_name = index if index == None else f"{index}_mult"
    skip_name = index if index == None else f"{index}_skip"
    res_name = index if index == None else f"{index}_res"
    # feed forward
    tanh_out = sigDCCLayer(
        x,
        n_filters,
        kernel_size,
        dilation_rate=dilation_rate,
        name=tanh_name,
        activation="tanh",
    )
    sigm_out = sigDCCLayer(
        x,
        n_filters,
        kernel_size,
        dilation_rate=dilation_rate,
        name=sig_name,
        activation="sigmoid",
    )
    out = Multiply(name=mult_name)([tanh_out, sigm_out])
    skip = sigDCCLayer(out, n_filters, 1, name=skip_name)
    res = Add(name=res_name)([skip, x])
    return res, skip


def compute_receptive_field(rf_in, cf_in, kern_sz, dil_rate):
    """Computes the receptive field and coverage factor of the residual block.

    # Arguments

        rf_in: integer representing the receptive field of
            the inputs to the layer.
        cf_in: float representing the coverage factor of the
            inputs to the layer.
        kern_sz: integer representing the kernel size of the
            layer.
        dil_rate: integer denoting the dilatation rate of the
            layer.

    # Returns
        
        rec_field: an integer denoting the receptive field of
            the layer.
        cov_factor: a float denoting the gap factor of the layer.

    # Note: rec_field tells how much of the original input contributes
        to the output node of each layer. The cov_factor tells what
        percentage of this original input is actually used.
    """
    rec_field = rf_in + (kern_sz - 1) * dil_rate
    if dil_rate <= rf_in:
        cov_factor = cf_in
    elif rf_in == 1:  # input layer
        cov_factor = 1.0 / float(dil_rate)
    else:
        print(
            "WARNING: Using large dilation rates resulting in partial-coverage receptive fields."
        )
        cov_factor = cf_in * (
            1.0 - float(kern_sz - 1) * float(dil_rate - rf_in) / float(rec_field)
        )
    return rec_field, cov_factor


def sigDCCResBlockStack(
    input_tensor,
    rb_dilation_rate=[1, 4, 16, 64, 192, 256],
    rb_filters=50,
    rb_kernel_size=6 * [4],
    post_activation="relu",
    rec_field = 1,
    cov_factor = 1.0,
    index="",
):
    """Stack successive sigDCCResidualBlocks with custom dilation rates and kernel sizes.

    # Arguments

        input_tensor: (keras tensor)
        rb_dilation_rate: (list of integers) dilation rates for the residual block stack.
        rb_kernel_size: (list of integers) kernel sizes for the residual block stack.
        rb_filters: (integer) filters per block in the sigDCC RB stack.
        post_activation: (string) activation function to use for each residual block.
        rec_field: (integer) the receptive field of the input tensor.
        cov_factor: (float in [0, 1.0]) the coverage factor of the input tensor.
        index: (string) for naming the layers in this stack.

    # Returns

        Keras tensor that is a concatenation of all residual blocks skip connections.
    """
    skip_connections = []

    # spot check input list lengths
    rb_parameters = [rb_dilation_rate, rb_kernel_size]
    rb_parameters_lengths = [len(ti) for ti in rb_parameters]
    if len(set(rb_parameters_lengths)) > 1:
        raise ValueError(
            "Expected all rb_parameters to be same length. Received {}".format(
                rb_parameters_lengths
            )
        )
    rb_dilation_depth = len(rb_dilation_rate)

    # build stack of residual blocks
    out = input_tensor
    for i in range(rb_dilation_depth):
        block_name = f"{index}_rb{i}"
        rec_field, cov_factor = compute_receptive_field(
            rec_field, cov_factor, rb_kernel_size[i], rb_dilation_rate[i]
        )
        out, skip = sigDCCResidualBlock(
            out,
            i,
            rec_field,
            cov_factor,
            kernel_size=rb_kernel_size[i],
            dilation_rate=rb_dilation_rate[i],
            n_filters=rb_filters,
            index=block_name,
        )
        skip_connections.append(skip)

    # concatentate skip connections
    name = "add_skip_connections" + index
    out = Add(name=name)(skip_connections)
    out = Activation(post_activation)(out)
    name = "skip_BN" + index
    out = BatchNormalization(name=name)(out)

    return out


def sigConv_and_pooling_block(
    x,
    filters,
    kernel_size,
    conv_act=None,
    name=None,
    padding="same",
    conv_depth=1,
    batch_norm=True,
    res_con=False,
    pooling_rate=4,
):
    """add a block of sigDCCLayers with dilation rate 1 followed by a pooling layer.

    # Arguments

        x: (keras tensor) input tensor
        filters: (integer) number of filters for the block convolutional layers.
        kernel_size: (integer) kernel size for the block convolutional layers.
        conv_act: (string) activation function for the block convolutional layers.
        name: (string) used for naming the layers.
        padding: (string) "causal" or "same"
        conv_depth: (integer) number of convolutions in the block.
        batch_norm: (boolean) whether to add batch normalization layers.
        res_con: (boolean) use residual connection. Requires block input shape and
            output shape to be the same.
        pooling_rate: (integer) the pooling rate to use for this block.

    # Returns

        The feature maps of the sigDCC convolution pooling block 
        with shape (batch, signal length, 2, filters).
    """
    res = x
    for i in range(conv_depth):
        x = sigDCCLayer(
            x,
            filters,
            kernel_size,
            name=name if name == None else f"{name}_conv_{i}",
            padding=padding,
            activation=conv_act,
        )
        if res_con:
            x = Add(name=name if name == None else f"{name}_rc_{i}")([x, res])
        if batch_norm:
            x = BatchNormalization(name=name if name == None else f"{name}_bn_{i}")(x)

    # pooling at the end of the block
    x = MaxPooling2D(
        pool_size=(pooling_rate, 1),
        name=name if name == None else f"{name}_maxpool_{i}",
    )(x)
    return x
