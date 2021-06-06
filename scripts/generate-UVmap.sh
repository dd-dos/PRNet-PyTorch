export PYTHONPATH="."
python3 data/processor.py -i=data/images/AFLW2000 -o=data/images/AFLW2000-crop -f=True -v=True --isOldKpt=True --thread=4
python3 data/processor.py -i=data/images/300W_LP -o=data/images/300W_LP-crop --thread=4