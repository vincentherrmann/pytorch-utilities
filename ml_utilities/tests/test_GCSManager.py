from unittest import TestCase
import google.cloud.storage as storage
import google.auth

class TestGCSManager(TestCase):
    def test_storageClient(self):
        credentials, project = google.auth.default()
        self.client = storage.client.Client('pytorch-wavenet', )
