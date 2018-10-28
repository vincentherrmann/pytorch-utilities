import tensorflow as tf
import numpy as np
import scipy.misc
import threading
from matplotlib import pyplot as plt
from IPython import display

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class TextLogger:
    def __init__(self,
                 log_interval=100,
                 validation_function=None,
                 validation_interval=1000,
                 snapshot_function=None,
                 snapshot_interval=1000,
                 background_function=None,
                 background_interval=1000):

        self.log_interval = log_interval
        self.validation_function = validation_function
        self.validation_interval = validation_interval
        self.snapshot_function = snapshot_function
        self.snapshot_interval = snapshot_interval
        self.background_function = background_function
        self.background_interval = background_interval

        self.accumulated_loss = 0
        if self.background_function is not None:
            self.background_thread = threading.Thread(target=self.background_function)
            self.background_thread.daemon = True

    def log(self, current_step, current_loss):
        self.accumulated_loss += current_loss
        if current_step % self.log_interval == 0:
            self.log_loss(current_step)
            self.accumulated_loss = 0
        if current_step % self.validation_interval == 0:
            if self.validation_function is not None:
                self.validate(current_step)
        if current_step % self.snapshot_interval == 0:
            if self.snapshot_function is not None:
                self.snapshot(current_step)
        if current_step % self.background_interval == 0:
            if self.background_function is not None:
                self.background(current_step)

    def log_loss(self, current_step):
        avg_loss = self.accumulated_loss / self.log_interval
        print("loss at step " + str(current_step) + ": " + str(avg_loss))

    def validate(self, current_step):
        avg_loss, avg_accuracy = self.validation_function()
        print("validation loss: " + str(avg_loss))
        print("validation accuracy: " + str(avg_accuracy * 100) + "%")

    def snapshot(self, current_step):
        self.snapshot_function(current_step)

    def background(self, current_step):
        if self.background_thread.is_alive():
            print("Previous background function is still running, skipping this one.")
        else:
            self.background_thread = threading.Thread(target=self.background_function,
                                                      args=[current_step])
            self.background_thread.daemon = True
            self.background_thread.start()


class JupyterLogger(TextLogger):
    def __init__(self,
                 log_interval=100,
                 validation_function=None,
                 validation_interval=1000,
                 snapshot_function=None,
                 snapshot_interval=1000,
                 background_function=None,
                 background_interval=1000):
        super().__init__(log_interval=log_interval,
                         validation_function=validation_function,
                         validation_interval=validation_interval,
                         snapshot_function=snapshot_function,
                         snapshot_interval=snapshot_interval,
                         background_function=background_function,
                         background_interval=background_interval)
        self.loss_steps = []
        self.loss_values = []
        self.validation_steps = []
        self.validation_losses = []
        self.validation_accuracies = []

    def log_loss(self, current_step):
        avg_loss = self.accumulated_loss / self.log_interval
        self.loss_steps.append(current_step)
        self.loss_values.append(avg_loss)
        self.draw()

    def validate(self, current_step):
        avg_loss, avg_accuracy = self.validation_function()
        self.validation_steps.append(current_step)
        self.validation_losses.append(avg_loss)
        self.validation_accuracies.append(avg_accuracy)
        self.draw()

    def draw(self):
        display.clear_output(wait=True)

        plt.plot(self.loss_steps, self.loss_values)
        plt.ylabel("train loss")
        plt.show()

        plt.plot(self.validation_steps, self.validation_losses)
        plt.ylabel("validation loss")
        plt.show()

        plt.plot(self.validation_steps, self.validation_accuracies)
        plt.ylabel("validation accuracy")
        plt.show()


class TensorboardLogger(TextLogger):
    def __init__(self,
                 log_interval=100,
                 validation_function=None,
                 validation_interval=1000,
                 snapshot_function=None,
                 snapshot_interval=1000,
                 background_function=None,
                 background_interval=1000,
                 log_directory='logs',
                 log_histograms=False,
                 model=None):
        super().__init__(log_interval=log_interval,
                         validation_function=validation_function,
                         validation_interval=validation_interval,
                         snapshot_function=snapshot_function,
                         snapshot_interval=snapshot_interval,
                         background_function=background_function,
                         background_interval=background_interval)
        self.tb_writer = tf.summary.FileWriter(log_directory)
        self.model = model
        self.log_histograms = log_histograms

    def log_loss(self, current_step):
        avg_loss = self.accumulated_loss / self.log_interval
        self.scalar_summary('loss', avg_loss, current_step)

    def validate(self, current_step):
        avg_loss, avg_accuracy = self.validation_function()
        self.scalar_summary('validation loss', avg_loss, current_step)
        self.scalar_summary('validation accuracy', avg_accuracy, current_step)

        # parameter histograms
        if not self.log_histograms or self.model is None:
            return
        for tag, value, in self.trainer.model.named_parameters():
            tag = tag.replace('.', '/')
            self.histo_summary(tag, value.data.cpu().numpy(), current_step)
            if value.grad is not None:
                self.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), current_step)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.tb_writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.tb_writer.add_summary(summary, step)

    def audio_summary(self, tag, sample, step, sr=16000):
        with tf.Session() as sess:
            audio_summary = tf.summary.audio(tag, sample, sample_rate=sr, max_outputs=4)
            summary = sess.run(audio_summary)
            self.tb_writer.add_summary(summary, step)
            self.tb_writer.flush()


    def histo_summary(self, tag, values, step, bins=200):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.tb_writer.add_summary(summary, step)
        self.tb_writer.flush()

    def tensor_summary(self, tag, tensor, step):
        tf_tensor = tf.Variable(tensor).to_proto()
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, tensor=tf_tensor)])
        #summary = tf.summary.tensor_summary(name=tag, tensor=tensor)
        self.tb_writer.add_summary(summary, step)