from io import BytesIO

import scipy.misc
import tensorflow as tf
import numpy as np


class Logger(object):

    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step)
        self.writer.flush()

    def image_summary(self, tag, image, step):
        s = BytesIO()
        image = (image*256).astype(int)
        scipy.misc.toimage(image).save(s, format="png")

        # Create an Image object
        with self.writer.as_default():
            tf.summary.image(s.getvalue(), image, step)
            # Create and write Summary
        self.writer.flush()

    def image_list_summary(self, tag, images, step):
        if len(images) == 0:
            return

        with self.writer.as_default():
            tf.summary.image("{}/{}".format(tag, len(images)), images, step)

        self.writer.flush()
