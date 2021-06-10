rclone copy gran:gran/training_data/HELEN.zip . -v
unzip HELEN.zip 
rm HELEN.zip

rclone copy gran:gran/training_data/LFPW.zip . -v
unzip LFPW.zip 
rm LFPW.zip

rclone copy gran:gran/training_data/AFW.zip . -v
unzip AFW.zip 
rm AFW.zip

rclone copy gran:gran/training_data/IBUG.zip . -v
unzip IBUG.zip
rm IBUG.zip

rclone copy gran:gran/training_data/AFLW2000.zip . -v
unzip AFLW2000.zip
rm AFLW2000.zip

pip3 install -r requirements.txt
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
clearml-init
dvc pull