try:
    from google.colab import auth
except:
    from google import auth
from google.cloud import storage
import subprocess
import os.path
import threading
import torch
import glob
from ml_utilities.pytorch_utilities import load_to_cpu


class GCSManager:
    def __init__(self, project_id, bucket):
        try:
            auth.authenticate_user()
        except:
            print("not on colab")
        self.storage_client = storage.Client(project_id)
        self.bucket = self.storage_client.get_bucket(bucket)
        self.gc_bucket = '' #'gs://' + bucket

    def download_files(self, files, target_location):
        i = 0
        if not os.path.exists(target_location):
            os.makedirs(target_location)
        while i < len(files):
            for _ in range(10):  # try ten times
                file = os.path.join(self.gc_bucket, files[i])
                file_name = os.path.basename(file)
                file_target = os.path.join(target_location, file_name)
                if os.path.isfile(file_target):
                    print('downloaded', file_name)
                    i += 1
                    break
                blob = self.bucket.blob(file)
                blob.download_to_filename(file_target)

    def download_files_from_directory(self, location, target_location, max_count=None, sort_key=lambda x: x.updated, reverse=True):
        if not os.path.exists(target_location):
            os.makedirs(target_location)

        blobs = self.bucket.list_blobs(prefix=location)
        blobs = sorted(list(blobs), key=sort_key, reverse=reverse)

        if max_count is None:
            max_count = len(blobs)
        max_count = min(max_count, len(blobs))

        for i in range(max_count):
            blob = blobs[i]
            name = os.path.basename(blob.name)
            file_name = os.path.join(target_location, name)
            blob.download_to_filename(file_name)
            print("downloaded", file_name)

    def download_files_in_background(self, files, target_location):
        download_thread = threading.Thread(target=self.download_files, args=[files, target_location])
        download_thread.start()

    def upload_files(self, files, target_location):
        i = 0
        while i < len(files):
            for _ in range(10):  # try ten times
                file = files[i]
                file_name = os.path.basename(file)
                file_target = os.path.join(self.gc_bucket, target_location, file_name)
                blob = self.bucket.blob(file_target)
                try:
                    blob.upload_from_filename(file)
                    print("uploaded", file_target)
                    break
                except:
                    print('error while uploading ' + file_name)

    def upload_files_in_background(self, files, target_location):
        download_thread = threading.Thread(target=self.upload_files, args=[files, target_location])
        download_thread.start()

    def upload_files_from_directory(self, location, target_location, max_count=None, sort_key=os.path.getmtime, reverse=True):
        files = glob.glob(os.path.join(location, '*'))
        files = sorted(list(files), key=sort_key, reverse=reverse)

        if max_count is None:
            max_count = len(files)
        max_count = min(max_count, len(files))

        for i in range(max_count):
            file = files[i]
            name = os.path.basename(file)
            file_name = os.path.join(self.gc_bucket, target_location, name)
            blob = self.bucket.blob(file_name)
            blob.upload_from_filename(file)
            print("uploaded", file_name)


class SnapshotManager():
    def __init__(self,
                 model,
                 gcs_manager,
                 name='model',
                 snapshot_location='snapshots',
                 logs_location=None,
                 gcs_snapshot_location=None,
                 gcs_logs_location=None,
                 use_only_state_dict=False,
                 load_to_cpu=False):
        self.model = model
        self._name = name
        self.snapshot_location = snapshot_location
        self.logs_location = logs_location
        self.gcs_manager = gcs_manager
        self.gcs_snapshot_location = snapshot_location if gcs_snapshot_location is None else gcs_snapshot_location
        self.gcs_logs_location = self.current_tb_location if gcs_logs_location is None else gcs_logs_location
        self.use_only_state_dict = use_only_state_dict
        self.load_to_cpu = load_to_cpu

        self.name = name

        if not os.path.isdir(self.snapshot_location):
            os.makedirs(self.snapshot_location)

        if not os.path.isdir(self.logs_location):
            os.makedirs(self.logs_location)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        if not os.path.isdir(self.current_tb_location):
            os.makedirs(self.current_tb_location)

    @property
    def current_tb_location(self):
        return os.path.join(self.logs_location, self.name)

    def make_snapshot(self, current_step):
        name = self.name + "_" + str(current_step)
        path = os.path.join(self.snapshot_location, name)
        if self.use_only_state_dict:
            torch.save(self.model.state_dict(), path)
        else:
            torch.save(self.model, path)

    def upload_latest_files(self):
        try:
            self.gcs_manager.upload_files_from_directory(self.snapshot_location, self.gcs_snapshot_location, max_count=1)
            if self.logs_location is not None:
                self.gcs_manager.upload_files_from_directory(self.current_tb_location, self.gcs_logs_location, max_count=1)
        except:
            print("unable to upload files")

    def download_latest_files(self):
        try:
            self.gcs_manager.download_files_from_directory(self.gcs_snapshot_location, self.snapshot_location, max_count=1)
            if self.logs_location is not None:
                self.gcs_manager.download_files_from_directory(self.gcs_logs_location, self.current_tb_location)
        except:
            print("unable to download files")

    def load_latest_snapshot(self):
        files = glob.glob(os.path.join(self.snapshot_location, '*'))
        files = sorted(list(files), key=os.path.getmtime, reverse=True)
        if self.name not in files[0]:
            raise Exception('No matching file found')
        if self.use_only_state_dict:
            self.model.load_state_dict(torch.load(files[0]))
        else:
            if self.load_to_cpu:
                self.model = load_to_cpu(files[0])
            else:
                self.model = torch.load(files[0])
        return self.model, files[0]



