import collections
import json
import os
import random
import re

from absl import app
from absl import flags
from absl import logging
import pandas as pd
import tensorflow as tf
import gzip
import shutil

def main():
  data_directory = "data/raw"
  url = "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Outdoors_v1_00.tsv.gz"
  fname = "amazon_reviews_us_Outdoors_v1_00.tsv.gz"
  file_hash = "95a8b6a5d4cd30b7c3a79dbafb88ea78"
  extracted_dir_name = "amazon_reviews_us_Outdoors_v1_00.tsv"
  if not tf.io.gfile.exists(data_directory):
    tf.io.gfile.makedirs(data_directory)
  path_to_zip = tf.keras.utils.get_file(
      fname=fname,
      origin=url,
      file_hash=file_hash,
      hash_algorithm="md5",
      extract=True,
      cache_dir=data_directory)
 
  extracted_file_dir = os.path.join(
      os.path.dirname(path_to_zip), extracted_dir_name)
  
  with gzip.open(path_to_zip, 'rb') as f_in:
    with open(extracted_file_dir, 'wb') as f_out:
      shutil.copyfileobj(f_in, f_out)

  print("extracted data dir: ", path_to_zip, extracted_file_dir)
  return extracted_file_dir

if __name__ == "__main__":
    main()
    print("done")

