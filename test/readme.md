MODEL_DIR: ./output_roberta_iPET_100_test

Example:
  python3 create_ipet_training_set_test.py \
    --logits_dir ${MODEL_DIR}${DIR_SUFFIX}/ \
    --output_dir ${MODEL_DIR}${DIR_SUFFIX}/next-gen-train-sets/ \
    --data_dir ${DATA_DIR} \
    --task_name ${TASK} \
    --lm_train_examples_per_label 10000 \
    --reduction mean \
    --num_examples ${NEW_EXAMPLES} \
    --logits_percentage 0.25


iPET(iteration 1):
  python create_ipet_training_set_test.py \
    --logits_dir ./output_roberta_iPET_100_test/yelp/ \
    --output_dir ./output_roberta_iPET_100_test/next-gen-train-sets/ \
    --data_dir ../text_classification_datasets/yelp_review_full_csv \
    --task_name yelp-full \
    --lm_train_examples_per_label 10000 \
    --reduction mean \
    --num_examples 100 \
    --logits_percentage 0.25