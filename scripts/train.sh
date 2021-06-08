python3 train.py \
--train-root "data/images/300W_LP-crop" \
--test-root "data/images/AFLW2000-crop" \
--batch-size 32 \
--test-size 64 \
--save-path "weights" \
--num-workers 8 \
--visualize-path "result-samples"