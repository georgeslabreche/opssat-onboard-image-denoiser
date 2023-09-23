#!/bin/bash

folder=$1

# Loop through each file in the folder
for file in "$folder"/*; do

    # Get the filename without the path
    filename=$(basename "$file")
    
    # Remove "img_msec_" from the filename
    filename=${filename//img_msec_/}
    
    # Remove "_2" from the filename
    filename=${filename//_2/}
    
    # Remove "_2_thumbnail" from the filename
    filename=${filename//_thumbnail/}
    
    # Rename the file
    mv "$file" "$folder/$filename"
done