#!/bin/bash

folder=$1

# Loop through each file in the folder
for file in "$folder"/*; do

    # Get the filename without the path
    filename=$(basename "$file")

    # Replace "thumbnail.jpeg" with "denoised.jpeg"
    filename=${filename//_thumbnail.jpeg/.denoised.jpeg}
    
    # Replace "thumbnail.original.jpeg" with "thumbnail.jpeg"
    filename=${filename//_thumbnail.original.jpeg/.jpeg}
    
    # Replace "thumbnail.original.noised.jpeg" with "thumbnail.noised.jpeg"
    filename=${filename//_thumbnail.original.noised.jpeg/.noised.jpeg}

    
    # Remove "img_msec_" from the filename
    filename=${filename//img_msec_/}
    
    # Remove "_2" from the filename
    filename=${filename//_2/}
    
    # Rename the file
    mv "$file" "$folder/$filename"
done