import tensorflow as tf
from src.losses import AsymmetricLoss
from src.config import IMG_SIZE, NUM_CLASSES


def build_model(num_classes=NUM_CLASSES):
    """
    DenseNet121 pretrained on ImageNet with 320x320 input.
    Base frozen by default — unfreeze externally for fine-tuning phases.
    Layer names match saved checkpoint: gap, head_bn, drop1,
    fc1, fc_bn, drop2, predictions.
    """
    base = tf.keras.applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input_image')
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
    x = tf.keras.layers.Dropout(0.5, name='drop1')(x)
    x = tf.keras.layers.Dense(512, activation='relu', name='fc1')(x)
    x = tf.keras.layers.BatchNormalization(name='fc_bn')(x)
    x = tf.keras.layers.Dropout(0.3, name='drop2')(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation='sigmoid',
        dtype='float32', name='predictions'
    )(x)

    return tf.keras.Model(inputs, outputs, name='CheXNet_320')


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=AsymmetricLoss(),
        metrics=[
            tf.keras.metrics.AUC(multi_label=True, name='auc'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy')
        ]
    )
    return model


def load_trained_model(path='outputs/weights/best_model_phaseC.keras'):
    return tf.keras.models.load_model(
        path,
        custom_objects={'AsymmetricLoss': AsymmetricLoss},
        compile=False
    )