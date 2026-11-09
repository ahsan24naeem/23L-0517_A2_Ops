"""Model builders: a from-scratch baseline CNN and transfer-learning models.

Every model takes raw RGB pixels in [0, 255] and does its own scaling inside the
graph, so the saved .keras file is self-contained for serving.
"""
from __future__ import annotations

import keras
from keras import layers

BACKBONES = {
    # name: (constructor, preprocessing expected by the ImageNet weights)
    "mobilenetv2": "MobileNetV2",
    "resnet50": "ResNet50",
}


def build_baseline_cnn(img_size: int = 224, dropout: float = 0.4) -> keras.Model:
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale_0_1")(inputs)
    for i, filters in enumerate([32, 64, 128, 256], start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def _resnet_preprocess(x):
    # caffe-style: RGB->BGR and ImageNet mean subtraction, no scaling
    import keras.ops as ops

    x = x[..., ::-1]
    mean = ops.convert_to_tensor([103.939, 116.779, 123.68], dtype=x.dtype)
    return x - mean


@keras.saving.register_keras_serializable(package="visionguard")
class ResNetPreprocess(layers.Layer):
    def call(self, x):
        return _resnet_preprocess(x)


def build_transfer_model(
    backbone: str = "mobilenetv2", img_size: int = 224, dropout: float = 0.3
) -> tuple[keras.Model, keras.Model]:
    """Returns (full_model, backbone_model). Backbone starts frozen."""
    inputs = keras.Input((img_size, img_size, 3), name="image")
    if backbone == "mobilenetv2":
        # MobileNetV2 was trained on inputs scaled to [-1, 1]
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mnv2_preprocess")(inputs)
        base = keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
        )
    elif backbone == "resnet50":
        x = ResNetPreprocess(name="resnet_preprocess")(inputs)
        base = keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
        )
    else:
        raise ValueError(f"unknown backbone {backbone}")
    base.trainable = False
    # training=False keeps BatchNorm layers in inference mode even after unfreezing,
    # so their ImageNet statistics are not destroyed by our small batches.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inputs, outputs, name=f"{backbone}_transfer"), base


# Index of the first layer of the "last block" for each backbone.
LAST_BLOCK_PREFIX = {"mobilenetv2": "block_16_", "resnet50": "conv5_block3_"}


def unfreeze_last_layers(base: keras.Model, backbone: str, n_layers: int | None = None) -> list[str]:
    """Unfreeze only the last block (default) or the last n_layers of the backbone.

    BatchNormalization layers stay frozen. Returns the names of trainable layers.
    """
    base.trainable = True
    if n_layers is None:
        prefix = LAST_BLOCK_PREFIX[backbone]
        start = next(i for i, l in enumerate(base.layers) if l.name.startswith(prefix))
    else:
        start = len(base.layers) - n_layers
    for i, layer in enumerate(base.layers):
        trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
        layer.trainable = trainable
    return [l.name for l in base.layers if l.trainable and l.weights]


def trainable_layer_names(model: keras.Model) -> list[str]:
    names = []
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            names += [f"{layer.name}/{l.name}" for l in layer.layers if l.trainable and l.weights]
        elif layer.trainable and layer.weights:
            names.append(layer.name)
    return names
