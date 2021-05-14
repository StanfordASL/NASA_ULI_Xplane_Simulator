# these are the URLS for data at the stanford website

download_dir=${NASA_DATA_DIR}/downsampled/
rm -rf ${download_dir}
mkdir -p ${download_dir}

cd ${download_dir}

# downsampled data:
for fname in overcast night morning afternoon;
do
    echo ${fname}
    wget https://stacks.stanford.edu/file/druid:zz143mb4347/${fname}_downsampled.zip | unzip {}

    unzip ${download_dir}/${fname}_downsampled.zip
    rm ${download_dir}/${fname}_downsampled.zip

done

