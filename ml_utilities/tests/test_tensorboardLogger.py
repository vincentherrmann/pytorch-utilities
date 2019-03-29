from unittest import TestCase
from ml_utilities.src.ml_utilities.train_logging import *

import shutil
import os.path
import numpy as np


class TestTensorboardLogger(TestCase):
    def setUp(self):
        if os.path.exists('test_logs'):
            shutil.rmtree('test_logs')
        self.logger = TensorboardLogger(log_directory='test_logs')

    def test_lossLogging(self):
        for step in range(10000):
            self.logger.log(step, np.exp(-step / 5000))

    def test_arrayData(self):
        data = np.exp(np.linspace(0, -2, num=12))
        for step in range(20):
            data += 0.1
            data *= 0.9
            self.logger.writer.add_histogram('prediction loss', data, step)

