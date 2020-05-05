###Dataset download

print('Beginning dataset download')
urllib.request.urlretrieve(url,download_directory + zipped_dataset_name)
print('Dataset download complete')

tar_file = tarfile.open(download_directory + zipped_dataset_name)
tar_file.extractall(download_directory + unzipped_dataset_name)
print('Dataset unzipped')