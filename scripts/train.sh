python3 train.py \
--train-root "data/images/300W_LP-crop" \
--test-root "data/images/AFLW2000-crop" \
--batch-size 128 \
--test-size 128 \
--save-path "weights" \
--num-workers 8 \
--pretrained "weights/2021-06-08/best.pth"