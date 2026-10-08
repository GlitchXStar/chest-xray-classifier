import tensorflow as tf


class AsymmetricLoss(tf.keras.losses.Loss):

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        label_smoothing=0.02,
        name='AsymmetricLoss'
    ):
        super().__init__(name=name)
        self.gamma_neg       = gamma_neg
        self.gamma_pos       = gamma_pos
        self.clip            = clip
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_true = y_true * (1.0 - self.label_smoothing) \
               + 0.5    *         self.label_smoothing

        y_pred_neg = tf.clip_by_value(y_pred - self.clip, 0.0, 1.0)
        y_pred     = tf.clip_by_value(y_pred,             1e-7, 1.0 - 1e-7)
        y_pred_neg = tf.clip_by_value(y_pred_neg,         1e-7, 1.0 - 1e-7)

        log_pos = tf.math.log(y_pred)
        log_neg = tf.math.log(1.0 - y_pred_neg)

        p_t_pos = y_pred
        p_t_neg = 1.0 - y_pred_neg
        w_pos   = tf.pow(1.0 - p_t_pos, self.gamma_pos)
        w_neg   = tf.pow(1.0 - p_t_neg, self.gamma_neg)

        loss = -(
            y_true         * w_pos * log_pos +
            (1.0 - y_true) * w_neg * log_neg
        )
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'gamma_neg':       self.gamma_neg,
            'gamma_pos':       self.gamma_pos,
            'clip':            self.clip,
            'label_smoothing': self.label_smoothing,
        })
        return cfg