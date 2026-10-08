import os
import tensorflow as tf
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')

from app import predict_base, explain_disease, CLASSES

print('Running test...')
dummy_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
results, update = predict_base(dummy_image)
disease = update['value']
print('Selected disease:', disease)

outputs = explain_disease(dummy_image, disease)
print('OUTPUTS LENGTH:', len(outputs))
for i, out in enumerate(outputs):
    if isinstance(out, np.ndarray):
        print(f'Output {i}: type={type(out)}, shape={out.shape}')
    else:
        print(f'Output {i}: {out}')

print('SUCCESS')
