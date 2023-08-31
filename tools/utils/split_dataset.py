#!/usr/bin/env python3

import os
import sys
import shutil
from sklearn.model_selection import train_test_split
import argparse

def split_dataset(input_dir, ratio, seed, train_dir, test_dir):
  """
  Splits the data in the input directory into a train and test set and copies
  them to their respective output folders.
  
  Parameters:
  - input_dir: The input directory containing all images.
  - ratio: The ratio in which data needs to be split. For example, if ratio = 0.8,
           80% data will be in training and 20% in test.
  - seed: Seed value for random splitting.
  - train_dir: Output folder for training data.
  - test_dir: Output folder for test data.
  """
  
  # list all files in the input directory
  files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
  
  # split the files into train and test using train_test_split
  train_files, test_files = train_test_split(files, train_size=ratio, random_state=seed)
  
  # ensure the output folders exist
  if not os.path.exists(train_dir):
      os.makedirs(train_dir)
  if not os.path.exists(test_dir):
      os.makedirs(test_dir)
  
  # copy the train files to the train output folder
  for f in train_files:
      shutil.copy2(os.path.join(input_dir, f), os.path.join(train_dir, f))
  
  # copy the test files to the test output folder
  for f in test_files:
      shutil.copy2(os.path.join(input_dir, f), os.path.join(test_dir, f))
      
  print(f"{(ratio)*100}% of the input data copied to: {train_dir}")
  print(f"{(1-ratio)*100}% of the input data copied to: {test_dir}")

def main():
  parser = argparse.ArgumentParser(description="Split dataset into train and test folders.")
  parser.add_argument("--input_dir", help="Path to the input directory containing images.")
  parser.add_argument("--ratio", type=float, default=0.9, help="Ratio to split dataset. Value between 0 and 1. Default is 0.9.")
  parser.add_argument("--seed", type=int, default=88, help="Seed value for random splitting. Default is 88.")
  parser.add_argument("--train_dir", help="Path to the output folder for training data.")
  parser.add_argument("--test_dir", help="Path to the output folder for test data.")

  # parse arguments
  args = parser.parse_args()
  
  # check if no arguments are provided and display help
  if len(sys.argv) == 1:
     parser.print_help(sys.stderr)
     sys.exit(1)

  # split the given datase
  split_dataset(args.input_dir, args.ratio, args.seed, args.train_dir, args.test_dir)

if __name__ == "__main__":
  main()
