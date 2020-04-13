from globals import BUCKET_NAME, LOCAL_BUCKET
import boto3
from io import BytesIO
import os
import pathlib


class Storage:
    def __init__(self, bucket=BUCKET_NAME):
        self._bucket_name = bucket
        self._bucket = self.get_bucket()

    def get_bucket(self):
        client = boto3.resource('s3')
        return client.Bucket(self._bucket_name)

    def put(self, key, value):
        return self._bucket.put_object(Key=key, Body=value)

    def get(self, key):
        io = BytesIO()
        self._bucket.download_fileobj(key, io)
        io.seek(0)
        return io.read()

    def delete(self, key):
        return self._bucket.delete_objects(
            Delete={
                'Objects': [
                    {
                        'Key': key
                    }
                ]})

    def ls(self):
        for obj in self._bucket.objects.all():
            yield obj.key

            
class FsStorage:
    def __init__(self, base_folder=LOCAL_BUCKET):
        self._prefix = base_folder

    def get_bucket(self):
        return self

    def put(self, key, value):
        if len(key.split('/')) > 1:
            path = "/".join(key.split("/")[:-1])
            pathlib.Path(self._prefix + path).mkdir(parents=True, exist_ok=True)
        with open(self._prefix + key, 'wb') as f:
            f.write(value)
    
    def get(self, key):
        with open(self._prefix + key, 'rb') as f:
            return f.read()

    def delete(self, key):
        return os.remove(self._prefix + key)

    def ls(self):
        return os.listdir(self._prefix)

def test():
    storage = Storage()
    storage.put('test', b'a')
    print(storage.get('test'))
    storage.delete('test')

