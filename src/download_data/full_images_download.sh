# these are the URLS for data at the stanford website

download_dir=${NASA_DATA_DIR}/large_images/
rm -rf ${download_dir}
mkdir -p ${download_dir}

cd ${download_dir}

# downsampled data:
for fname in overcast night morning afternoon;
do
    echo ${fname}
    wget https://stacks.stanford.edu/file/druid:zz143mb4347/${fname}.zip | unzip {}

    unzip ${download_dir}/${fname}.zip
    rm ${download_dir}/${fname}.zip

done

