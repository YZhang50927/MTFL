#!/bin/bash

# Define the source directory and the destination directories
source_dir1="/media/mount_loc/yiling/UCF_Crimes"
source_dir2="/media/mount_loc/yiling/Throwing"
source_dir3="/media/mount_loc/yiling/Accident"
destination_dir="/media/mount_loc/yiling/3in1"

# define the classification source file for RoadAccidents and Throwing
source_file_UCFR="/home/yiling/workspace/data_preparation/reclassification/BodyExposedAccidentsInUCF.txt"
source_file_TC="/home/yiling/workspace/data_preparation/reclassification/ThrowingObjectsAtCar.txt"

# Create the destination directories
mkdir -p "${destination_dir}/PropertyCrime"
mkdir -p "${destination_dir}/Violence"
mkdir -p "${destination_dir}/BodyExposedAccidents"
mkdir -p "${destination_dir}/EnclosedVehicleAccidents"
mkdir -p "${destination_dir}/Littering"
mkdir -p "${destination_dir}/Normal"

mkdir -p "${destination_dir}/PropertyCrime/ThrowingObjectsAtCar"
mkdir -p "${destination_dir}/Violence/ThrowingObjectsAtSomeone"
mkdir -p "${destination_dir}/BodyExposedAccidents/UCFRoadAccidents"
mkdir -p "${destination_dir}/EnclosedVehicleAccidents/UCFRoadAccidents"

# Move UCF to the appropriate destination directories based on their file types
for directory in "${source_dir1}"/*; do
    if [ -d "${directory}" ]; then
        case "${directory}" in
            *Shoplifting|*Stealing|*Vandalism|*Burglary|*Arson)
                mv "${directory}" "${destination_dir}/PropertyCrime/"
                echo "Moved : ${directory} to : ${destination_dir}/PropertyCrime/"
                ;;
            *Abuse|*Assault|*Fighting|*Robbery|*Arrest|*Shooting)
                mv "${directory}" "${destination_dir}/Violence/"
                echo "Moved : ${directory} to : ${destination_dir}/Violence/"
                ;;
            *Explosion)
                mv "${directory}" "${destination_dir}/"
                echo "Moved : ${directory} to : ${destination_dir}/"
                ;;
            *RoadAccidents)
                while IFS= read -r file; do
                    if [ -f "${directory}/${file}" ]; then
                        mv "${directory}/${file}" "${destination_dir}/BodyExposedAccidents/UCFRoadAccidents/"
                        echo "Moved file: $file"
                    else
                        echo "File not found: $file"
                    fi
                done < "${source_file_UCFR}"
                mv "${directory}"/* "${destination_dir}/EnclosedVehicleAccidents/UCFRoadAccidents"
                echo "Moved file: ${directory}/* to EnclosedVehicleAccidents/UCFRoadAccidents"
                ;;
        esac
    fi
done
# Move Throwing to the appropriate destination directories based on their file types
while IFS= read -r file; do
    find "${source_dir2}"/* -type f -name "${file}" -exec mv {} "${destination_dir}/PropertyCrime/ThrowingObjectsAtCar/" \;
    echo "Moved file: $file"
done < "${source_file_TC}"
find "${source_dir2}"/* -type f -exec mv {} "${destination_dir}/Violence/ThrowingObjectsAtSomeone/" \;
echo "Moved file: ${source_dir2}/* to /Violence/ThrowingObjectsAtSomeone/"

