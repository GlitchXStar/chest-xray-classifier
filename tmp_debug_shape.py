import tensorflow as tf
from src.losses import AsymmetricLoss
from src.config import CLASSES
model = tf.keras.models.load_model('outputs/weights/best_model_phaseC.keras', custom_objects={'AsymmetricLoss': AsymmetricLoss}, compile=False)
densenet = next((l for l in model.layers if 'densenet' in l.name.lower()), None)
LAST_CONV = next((l.name for l in reversed(densenet.layers) if isinstance(l, tf.keras.layers.Conv2D)), None)
print("densenet output type:", type(densenet.output))
dn_out = densenet.output[0] if isinstance(densenet.output, list) else densenet.output
lc_out = densenet.get_layer(LAST_CONV).output
lc_out = lc_out[0] if isinstance(lc_out, list) else lc_out

sub_densenet = tf.keras.Model(inputs=densenet.input, outputs=[lc_out, dn_out])
print("sub_densenet outputs:", sub_densenet.output_shape)

conv_feats, base_feats = sub_densenet(tf.ones((1, 320, 320, 3)))
print("conv_feats shape:", conv_feats.shape)
print("base_feats shape:", base_feats.shape)

feat_shape = dn_out.shape[1:]
print("feat_shape:", feat_shape)

head_input = tf.keras.Input(shape=feat_shape, name='head_input')
x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(head_input)
print("gap output shape:", x.shape)
