export PYTHONPATH="."
python3 data/processor.py -i=data/images/AFLW2000 -o=data/images/AFLW2000-crop --isOldKpt=True --thread=0
# python3 data/processor-zip.py -i=data/images/300W_LP -o=data/images/300W_LP-crop --thread=1