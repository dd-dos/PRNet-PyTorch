export PYTHONPATH="."
python data/processor.py -i=data/images/AFLW2000 -o=data/images/AFLW2000-crop -f=True -v=True --isOldKpt=True --thread=8
python data/processor.py -i=data/images/300W_LP -o=data/images/300W_LP-crop --thread=8