export PYTHONPATH="."
python models/torchrun.py -train=False -test=True -pd=data/images/AFLW2000-crop -visualize=True --loadModelPath=weights/best.pth 