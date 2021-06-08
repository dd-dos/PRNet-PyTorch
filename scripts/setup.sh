mkdir data/images
mkdir data/images/300WLP
cd data/images

rclone copy gran:gran/training_data/HELEN.zip . -v
unzip HELEN.zip -d 300WLP
rm HELEN.zip

rclone copy gran:gran/training_data/LFPW.zip . -v
unzip LFPW.zip -d 300WLP
rm LFPW.zip

rclone copy gran:gran/training_data/AFW.zip . -v
unzip AFW.zip -d 300WLP
rm AFW.zip

rclone copy gran:gran/training_data/IBUG.zip . -v
unzip IBUG.zip -d 300WLP
rm IBUG.zip

rclone copy gran:gran/training_data/AFLW2000.zip . -v
unzip AFLW2000.zip
rm AFLW2000.zip