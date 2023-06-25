#!/usr/bin/env python3

from radiant_mlhub import Collection
from radiant_mlhub import Dataset
from dateutil.parser import parse

from pprint import pprint
from constants import *


def download_imagery_all(collection_id):
  ''' download all imagery from a collection (because we can't get filtering to work...) '''

  # get collection for landsat imagery
  collection = Collection.fetch(collection_id)

  # print archive size
  print(collection.archive_size)

  # download the imagery archive
  collection.download(DIR_PATH_IMAGES_DOWNLOAD)


def download_imagery_filtered(dataset_id, imagery_source):
  ''' this doesn't work '''

  # collection imagery id
  collection_imagery_id = f'{dataset_id}_source_{imagery_source}'

  # collection labels id
  #collection_labels_id = f'{dataset_id}_labels'

  # fetch the dataset
  dataset = Dataset.fetch(dataset_id)
  print('Fetched dataset: ', dataset)

  # the collection
  collection = None
  found_collection = False

  # fetch the target collection
  for collection in dataset.collections:
    if collection.id == collection_imagery_id:
      found_collection = True
      break

  # check if collection has been found
  if not found_collection:
    print(f'Collection not found: {collection_imagery_id}')
    exit(1)
  else:
    print(f'Found collection: {collection_imagery_id}')

  # date range
  #start_date = parse("2019-04-01T00:00:00+0000")
  #end_date = parse("2019-04-02T00:00:00+0000")

  # the filter
  # only fetch bands for Red, Green, and Blue (RGB)
  # don't fetch any labels
  collection_filter = {}
  collection_filter[collection_imagery_id] = ['B04', 'B03', 'B02']

  # download the data
  dataset.download(
    #datetime          = (start_date, end_date),
    collection_filter = collection_filter,
    output_dir        = DIR_PATH_IMAGES_DOWNLOAD)



# list all collections
'''
collections = Collection.list()
for c in collections:
  print(c)
'''

# list all datasets
'''
datasets = Dataset.list()
for d in datasets:
  print(d)
'''



# select the dataset id to download imagery
dataset_id = LANDCOVERNET_DATASET_ID_NA

# the source of the imagery
imagery_source = LANDCOVERNET_IMAGERY_SOURCE_LANDSAT8

# collection imagery id
collection_imagery_id = f'{dataset_id}_source_{imagery_source}'

# collection labels id
collection_labels_id = f'{dataset_id}_labels'

# download the imagery
# FIXME: download doesn't start for Europe dataset (ref_landcovernet_eu_v1)
download_imagery_all(collection_imagery_id)

# this does not work
# download_imagery_filtered(dataset_id, imagery_source)