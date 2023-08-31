#!/usr/bin/env python3

import os
import sys
import argparse
import shutil

def copy_matching_files(dir1, dir2, output_dir):
  """
  Copies files from dir2 to output_dir if they also exist in dir1.
  Prints a message for every file in dir2 that doesn't exist in dir1.

  Parameters:
  - dir1: Directory to check for filenames.
  - dir2: Directory to copy files from.
  - output_dir: Directory to copy files to.
  """

  # Ensure the output directory exists
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # Get list of filenames in dir1
  filenames_in_dir1 = set(os.listdir(dir1))
  filenames_in_dir2 = set(os.listdir(dir2))

  for filename in filenames_in_dir2:
    source_path = os.path.join(dir2, filename)
    destination_path = os.path.join(output_dir, filename)

    if filename in filenames_in_dir1:
      shutil.copy2(source_path, destination_path)
      print(f"Copied {filename}\n  from {dir2}\n  to {output_dir}\n")

def main():
  parser = argparse.ArgumentParser(description="Copy files from dir2 to output_dir if they exist in dir1. Print a message for unmatched files.")
  parser.add_argument("--dir1", help="Directory to check for filenames.")
  parser.add_argument("--dir2", help="Directory to copy files from.")
  parser.add_argument("--output_dir", help="Directory to copy files to.")

  args = parser.parse_args()

  # check if no arguments are provided and display help
  if len(sys.argv) == 1:
     parser.print_help(sys.stderr)
     sys.exit(1)

  copy_matching_files(args.dir1, args.dir2, args.output_dir)

if __name__ == "__main__":
    main()
