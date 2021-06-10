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