from io import BytesIO

import scipy.misc
import tensorflow as tf
import numpy as np

def gray2rgb(image):
    w, h = image.shape
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image * 255
    return ret

def outline(image, mask, color):
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
            image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    return image

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

    def log_images(self, x, y_true, y_pred, channel=0):
        images = []
        # print("x shape = " + str(x.shape))
        # print("y_true shape = " + str(y_true.shape))
        # print("y_pred shape = " + str(y_pred.shape))
        x_np = x[:, channel].cpu().numpy()
        y_true_np = y_true.cpu().numpy() #[:, 0].cpu().numpy()
        y_pred_np = y_pred.cpu().numpy() #[:, 0].cpu().numpy()
        # print("x_np shape = " + str(x_np.shape))
        # print("y_true_np shape = " + str(y_true_np.shape))
        # print("y_pred_np shape = " + str(y_pred_np.shape))
        for i in range(x_np.shape[0]):
            image = gray2rgb(np.squeeze(x_np[i]))
            image = outline(image, y_true_np[i], color=[0, 255, 0])
            for ch in range(y_pred_np.shape[1]):
                image = outline(image, y_pred_np[i,ch], color=[255, 0, 0])
            images.append(image)
        return images

    def image_list_summary(self, tag, images, step):
        if len(images) == 0:
            return

        with self.writer.as_default():
            tf.summary.image("{}/{}".format(tag, len(images)), images, step)

        self.writer.flush()
