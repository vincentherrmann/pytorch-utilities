from google.colab import files
from google.colab import auth
from oauth2client.client import GoogleCredentials
from operator import itemgetter
import subprocess
import os.path
import threading


class GCSManager():
    def __init__(self, project_id, bucket):
        auth.authenticate_user()
        self.project_id = project_id
        self.gc_bucket = 'gs://' + bucket
        subprocess.call(['gcloud', 'config', 'set', 'project', project_id])

    def download_files(self, files, target_location):
        i = 0
        returncode = 0
        if not os.path.exists(target_location):
            os.makedirs(target_location)
        while i < len(files):
            for _ in range(10):  # try ten times
                file = os.path.join(self.gc_bucket, files[i])
                file_name = os.path.basename(file)
                file_target = os.path.join(target_location, file_name)
                if os.path.isfile(file_target):
                    print('finished downloading', file_name)
                    i += 1
                    break
                elif returncode < 0:
                    print('error while downloading ' + file_name + ': ' + str(returncode))
                arguments = ['gsutil', 'cp', file, file_target]
                print("download", file, "to", file_target)
                returncode = subprocess.call(arguments)

    def download_files_in_background(self, files, target_location):
        download_thread = threading.Thread(target=self.download_files, args=[files, target_location])
        download_thread.start()

    def upload_files(self, files, target_location):
        i = 0
        returncode = 0
        while i < len(files):
            for _ in range(10):  # try ten times
                file = files[i]
                file_name = os.path.basename(file)
                file_target = os.path.join(self.gc_bucket, target_location, file_name)

                arguments = ['gsutil', 'cp', file, file_target]
                try:
                    returncode = subprocess.call(arguments)
                    if returncode >= 0:
                        print('finished uploading', file_name)
                        i += 1
                        break
                except:
                    print('error while uploading ' + file_name + ': ' + str(returncode))

    def upload_files_in_background(self, files, target_location):
        download_thread = threading.Thread(target=self.upload_files, args=[files, target_location])
        download_thread.start()

