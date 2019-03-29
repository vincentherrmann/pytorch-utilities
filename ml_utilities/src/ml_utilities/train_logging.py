from tensorboardX import SummaryWriter
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

        self.loss_meter = AverageMeter()
        if self.background_function is not None:
            self.background_thread = threading.Thread(target=self.background_function)
            self.background_thread.daemon = True

    def log(self, current_step, current_loss=None):
        if current_loss is not None:
            self.loss_meter.update(current_loss)
        if current_step % self.log_interval == 0:
            self.log_loss(current_step)
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
        print("loss at step " + str(current_step) + ": " + str(self.loss_meter.avg))
        self.loss_meter.reset()

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
        self.loss_steps.append(current_step)
        self.loss_values.append(self.loss_meter.avg)
        self.draw()
        self.loss_meter.reset()

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
                 log_directory=None,
                 log_histograms=False,
                 model=None):
        super().__init__(log_interval=log_interval,
                         validation_function=validation_function,
                         validation_interval=validation_interval,
                         snapshot_function=snapshot_function,
                         snapshot_interval=snapshot_interval,
                         background_function=background_function,
                         background_interval=background_interval)
        self.writer = SummaryWriter(log_directory)
        self.model = model
        self.log_histograms = log_histograms

    def log_loss(self, current_step):
        self.writer.add_scalar('loss', self.loss_meter.avg, current_step)
        self.loss_meter.reset()

    def validate(self, current_step):
        avg_loss, avg_accuracy = self.validation_function()
        self.writer.add_scalar('validation loss', avg_loss, current_step)
        self.writer.add_scalar('validation accuracy', avg_accuracy, current_step)

        # parameter histograms
        if not self.log_histograms or self.model is None:
            return
        for tag, value, in self.trainer.model.named_parameters():
            tag = tag.replace('.', '/')

            self.histo_summary(tag, value.data.cpu().numpy(), current_step)
            if value.grad is not None:
                self.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), current_step)

    def log_histogram(self, tag, values, step, bins=200):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)
        self.writer.add_histogram(tag, counts, step)


# from pytorch imagenet example
class AverageMeter(object):
    """Computes and stores the average, max and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.max = -1e38
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        if val > self.max:
            self.max = val
        self.count += n
        self.avg = self.sum / self.count

