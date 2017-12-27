savedir="model_data_keras2"
mkdir -p -- "$savedir"
export http_proxy="http://127.0.0.1:8123"
export https_proxy="http://127.0.0.1:8123"
wget https://www.dropbox.com/s/z7ittwfxa5css7a/model_data_keras2.zip?dl=0 -O $savedir/model_data_keras2.zip
unzip -j $savedir/model_data_keras2.zip -d $savedir
rm $savedir/model_data_keras2.zip
