import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config


def sync_s3_anonymous(bucket_name, prefix, local_dir):
    # Configure boto3 for public/anonymous access (--no-sign-request)
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    paginator = s3_client.get_paginator("list_objects_v2")

    print(f"Syncing s3://{bucket_name}/{prefix} to {local_dir}...")

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            s3_key = obj["Key"]

            # Skip directory marker objects
            if s3_key.endswith("/"):
                continue

            # Map S3 key hierarchy to local path
            relative_path = os.path.relpath(s3_key, prefix)
            local_file_path = os.path.normpath(
                os.path.join(local_dir, relative_path)
            )

            # Ensure local subdirectories exist before downloading
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            print(f"Downloading: {s3_key} -> {local_file_path}")
            s3_client.download_file(bucket_name, s3_key, local_file_path)


if __name__ == "__main__":
    BUCKET = "boreas"
    PREFIX = "boreas-2025-07-18-10-33"

    # Define your local $root path here
    ROOT_DIR = "/scratch/pg06/vm4619/boreas"  # e.g., '/home/user/data'
    LOCAL_DEST = os.path.join(ROOT_DIR, PREFIX)

    sync_s3_anonymous(BUCKET, PREFIX, LOCAL_DEST)