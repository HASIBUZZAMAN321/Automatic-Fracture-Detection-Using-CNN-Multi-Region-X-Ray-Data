import tensorflow as tf
from tensorflow.keras import layers, Model

def se_block(x, reduction=16, name="se"):
    c = int(x.shape[-1])
    r = max(c // reduction, 1)
    z = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    z = layers.Dense(r, activation="relu", name=f"{name}_fc1")(z)
    z = layers.Dense(c, activation="sigmoid", name=f"{name}_fc2")(z)
    z = layers.Reshape((1,1,c), name=f"{name}_reshape")(z)
    return layers.Multiply(name=f"{name}_scale")([x,z])

def channel_attention(x, reduction=16, name="ca"):
    c = int(x.shape[-1])
    r = max(c // reduction, 1)
    fc1 = layers.Dense(r, activation="relu", name=f"{name}_fc1")
    fc2 = layers.Dense(c, name=f"{name}_fc2")
    a = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    m = layers.GlobalMaxPooling2D(name=f"{name}_gmp")(x)
    a = fc2(fc1(a)); m = fc2(fc1(m))
    w = layers.Activation("sigmoid", name=f"{name}_sigmoid")(layers.Add()([a,m]))
    w = layers.Reshape((1,1,c), name=f"{name}_reshape")(w)
    return layers.Multiply(name=f"{name}_scale")([x,w])

def spatial_attention(x, kernel_size=7, name="sa"):
    avg = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
                        name=f"{name}_avg")(x)
    mx  = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
                        name=f"{name}_max")(x)
    z = layers.Concatenate(axis=-1, name=f"{name}_concat")([avg,mx])
    w = layers.Conv2D(1, kernel_size, padding="same", activation="sigmoid",
                      name=f"{name}_conv")(z)
    return layers.Multiply(name=f"{name}_scale")([x,w])

def cbam(x, reduction=16, spatial_kernel=7, name="cbam"):
    x = channel_attention(x, reduction, name=f"{name}_channel")
    x = spatial_attention(x, spatial_kernel, name=f"{name}_spatial")
    return x

def projected_skip(x, filters, pool_size, name):
    x = layers.Conv2D(filters, 1, padding="same", name=f"{name}_conv1x1")(x)
    return layers.MaxPooling2D(pool_size, name=f"{name}_pool")(x)

def build_model(
    input_shape=(128,128,3),
    dense_units=1024,
    conv_dropout=0.3,
    fc_dropout=0.5,
    cbam_spatial_kernel=7,
):
    """
    Paper-grounded reconstruction.

    Ambiguities preserved as parameters:
    - conv_dropout: paper reports 0.3 in Results.
    - fc_dropout: paper reports 0.5 after fully-connected layers in Discussion.
    - cbam_spatial_kernel: Eq. (3) says 3x3; detailed CBAM text/equation later says 7x7.
    - dense_units is not specified by the paper.
    """
    inp = layers.Input(input_shape, name="input_xray")

    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv32_1")(inp)
    x = layers.Dropout(conv_dropout, name="dropout32")(x)
    x = layers.MaxPooling2D(2, name="pool32_1")(x)
    x = layers.Conv2D(32, 2, padding="same", activation="relu", name="conv32_2")(x)
    x = layers.MaxPooling2D(2, name="pool32_2")(x)
    se = se_block(x, 16, "se32")

    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv64_1")(se)
    x = layers.Dropout(conv_dropout, name="dropout64")(x)
    x = layers.MaxPooling2D(2, name="pool64_1")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv64_2")(x)
    x = layers.MaxPooling2D(2, name="pool64_2")(x)
    att = cbam(x, 16, cbam_spatial_kernel, "cbam64")

    skip1 = projected_skip(se, 64, (4,4), "skip_se_to_cbam")
    x = layers.ReLU(name="relu_res64")(layers.Add(name="add_res64")([att,skip1]))
    cbam_out = x

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv128")(x)
    x = layers.MaxPooling2D(2, name="pool128")(x)
    sa = spatial_attention(x, 7, "spatial128")

    skip2 = projected_skip(cbam_out, 128, (2,2), "skip_cbam_to_spatial")
    x = layers.ReLU(name="relu_res128")(layers.Add(name="add_res128")([sa,skip2]))

    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="conv256")(x)
    x = layers.MaxPooling2D(2, name="pool256")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense")(x)
    x = layers.Dropout(fc_dropout, name="dropout_fc")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    return Model(inp, out, name="proposed_fracture_cnn")
