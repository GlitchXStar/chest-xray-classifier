import sys
import tensorflow as tf
from src.config import CLASSES
from src.losses import AsymmetricLoss
from src.uncertainty import perform_mc_dropout

model = tf.keras.models.load_model('outputs/weights/best_model_phaseC.keras', custom_objects={'AsymmetricLoss': AsymmetricLoss}, compile=False)
densenet = next((l for l in model.layers if 'densenet' in l.name.lower()), None)
LAST_CONV = next((l.name for l in reversed(densenet.layers) if isinstance(l, tf.keras.layers.Conv2D)), None)
sub_densenet = tf.keras.Model(inputs=densenet.input, outputs=[densenet.get_layer(LAST_CONV).output, densenet.output])

feat_shape = densenet.output_shape[1:]
head_input = tf.keras.Input(shape=feat_shape, name='head_input')
x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(head_input)
x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
x = tf.keras.layers.Dropout(0.5, name='drop1')(x)
x = tf.keras.layers.Dense(512, activation='relu', name='fc1')(x)
x = tf.keras.layers.BatchNormalization(name='fc_bn')(x)
x = tf.keras.layers.Dropout(0.3, name='drop2')(x)
preds = tf.keras.layers.Dense(14, activation='sigmoid', name='predictions')(x)
head_model = tf.keras.Model(inputs=head_input, outputs=preds)

img_tensor = tf.ones((1, 320, 320, 3))
mc_cams, mc_probs = perform_mc_dropout(sub_densenet, head_model, img_tensor, 0, passes=1)
print("SUCCESS!")
