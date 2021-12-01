#!/bin/bash
#SBATCH -A ashishmenon
#SBATCH -n 12
#SBATCH --nodelist=gnode87
#SBATCH --wait-all-nodes=0
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END



module load cuda/10.0
module load cudnn/7.3-cuda-10.0



CODE_path='/home/ashishmenon/SMILY++/pyfiles/Metric_learning_Interactive_ret_CRC/acpr_codebase'
DOWNLOAD_path='/ssd_scratch/cvit/ashishmenon'

mkdir -p ${CODE_path}
mkdir -p ${DOWNLOAD_path}

# cd ${DOWNLOAD_path}


# wget -O NCT-CRC-HE-100K.zip  https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip?download=1
# wget -O CRC-VAL-HE-7K.zip https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip?download=1
# wget -O ICIAR2018_BACH_Challenge.zip https://zenodo.org/record/3632035/files/ICIAR2018_BACH_Challenge.zip?download=1

# mkdir -p ${DOWNLOAD_path}/COLORECTAL_train_val_test/query
# mkdir -p ${DOWNLOAD_path}/COLORECTAL_train_val_test/train/
# mkdir -p ${DOWNLOAD_path}/COLORECTAL_train_val_test/test/
# mkdir -p ${DOWNLOAD_path}/COLORECTAL_train_val_test/annotated/

# unzip NCT-CRC-HE-100K.zip 
# unzip CRC-VAL-HE-7K.zip 


# mv NCT-CRC-HE-100K ./COLORECTAL_train_val_test/train/
# mv CRC-VAL-HE-7K ./COLORECTAL_train_val_test/test/

# cd COLORECTAL_train_val_test/query
# mkdir ADI  BACK  DEB  LYM  MUC  MUS  NORM  STR  TUM

# cd ../annotated/
# mkdir ADI  BACK  DEB  LYM  MUC  MUS  NORM  STR  TUM

cd ${CODE_path}/


# python dataset_rearrange.py --train_path ${DOWNLOAD_path}/COLORECTAL_train_val_test/train/NCT-CRC-HE-100K/ \




# python feature_database.py --dataroot ${DOWNLOAD_path}/COLORECTAL_train_val_test/train/NCT-CRC-HE-100K/ \
# 							--save_dir ${DOWNLOAD_path}/CRC_resnet_pretrained_features_train/ \
# 							--label_pos 2 \
# 							--use_fc 0 \
# 							--dataset CRC \
# 							--label_set ADI  BACK  DEB  LYM  MUC  MUS  NORM  STR  TUM

# python feature_database.py --dataroot ${DOWNLOAD_path}/COLORECTAL_train_val_test/query/  \
# 							--save_dir  ${DOWNLOAD_path}/CRC_resnet_pretrained_features_query/  \
# 							--label_pos 2 \
# 							--use_fc 0 \
# 							--dataset CRC \
# 							--label_set ADI  BACK  DEB  LYM  MUC  MUS  NORM  STR  TUM

# python feature_database.py --dataroot  ${DOWNLOAD_path}/COLORECTAL_train_val_test/annotated/  \
# 							--save_dir  ${DOWNLOAD_path}/CRC_resnet_pretrained_features_annotated/  \
# 							--label_pos 2 \
# 							--use_fc 0 \
# 							--dataset CRC \
# 							--label_set ADI  BACK  DEB  LYM  MUC  MUS  NORM  STR  TUM

# python feature_database.py --dataroot ${DOWNLOAD_path}/COLORECTAL_train_val_test/test/CRC-VAL-HE-7K/ \
# 							--save_dir ${DOWNLOAD_path}/CRC_resnet_pretrained_features_test/ \
# 							--label_pos 2 \
# 							--use_fc 0 \
# 							--dataset CRC \
# 							--label_set ADI  BACK  DEB  LYM  MUC  MUS  NORM  STR  TUM

k=10

search_dir="${DOWNLOAD_path}/CRC_resnet_pretrained_features_train"
query_dir="${DOWNLOAD_path}/CRC_resnet_pretrained_features_query"
annotated_dir="${DOWNLOAD_path}/CRC_resnet_pretrained_features_annotated"
annotated_img_dir="/ssd_scratch/cvit/ashishmenon/COLORECTAL_train_val_test/annotated"
test_dir="${DOWNLOAD_path}/CRC_resnet_pretrained_features_test"
csv_model_save_dir="${DOWNLOAD_path}/CRC_preds_resnet18"
log_dir="${DOWNLOAD_path}/tensorboard_CRC_preds_resnet18"




python -W ignore CRC_expt.py --save_results_name CRC_preds_resnet18_hybrid_${k}.csv \
													    --search_dir ${search_dir} \
													    --query_dir ${query_dir} \
													    --annotated_dir ${annotated_dir} \
													    --annotated_img_dir ${annotated_img_dir} \
													    --test_dir ${test_dir} \
													    --num_img_to_rev ${k} \
				                                        --use_aux_query 1 \
				                                        --use_query_refinement 1 \
				                                        --ret_strategy "front_mid_end_ret" \
				                                        --save_prediction_csv ${csv_model_save_dir}/hybrid_${k}/ \
				                                        --log_dir ${log_dir}/hybrid_${k}/ \
				                                        --label_pos 2 \
				                                        --classifier_pred_top_mid_end 1\
				                                        --label_set ADI  BACK  DEB  LYM  MUC  MUS  NORM  STR  TUM \
				                                        --model_save_path ${csv_model_save_dir}/hybrid_${k}/model_ckpt/


python -W ignore CRC_expt.py --save_results_name CRC_results_resnet18_entropy_based_${k}.csv \
													    --search_dir ${search_dir} \
													    --query_dir ${query_dir} \
													    --annotated_dir ${annotated_dir} \
													    --annotated_img_dir ${annotated_img_dir} \
													    --test_dir ${test_dir} \
													    --num_img_to_rev ${k} \
				                                        --use_aux_query 1 \
				                                        --use_query_refinement 0 \
				                                        --ret_strategy "front_mid_end_ret" \
				                                        --save_prediction_csv ${csv_model_save_dir}/entropy_based_${k}/ \
				                                        --log_dir ${log_dir}/entropy_based_${k}/ \
				                                        --label_pos 2 \
				                                        --entropy_based 1\
				                                        --label_set ADI  BACK  DEB  LYM  MUC  MUS  NORM  STR  TUM \
				                                        --model_save_path ${csv_model_save_dir}/entropy_based_${k}/model_ckpt/


python -W ignore CRC_expt.py --save_results_name CRC_preds_resnet18_CNFP_${k}.csv \
													    --search_dir ${search_dir} \
													    --query_dir ${query_dir} \
													    --annotated_dir ${annotated_dir} \
													    --annotated_img_dir ${annotated_img_dir} \
													    --test_dir ${test_dir} \
													    --num_img_to_rev ${k} \
				                                        --use_aux_query 1 \
				                                        --use_query_refinement 1 \
				                                        --ret_strategy "front_mid_end_ret" \
				                                        --save_prediction_csv ${csv_model_save_dir}/CNFP_${k}/ \
				                                        --log_dir ${log_dir}/CNFP_${k}/ \
				                                        --label_pos 2 \
				                                        --only_classifier_pred 1\
				                                        --label_set ADI  BACK  DEB  LYM  MUC  MUS  NORM  STR  TUM \
				                                        --model_save_path ${csv_model_save_dir}/CNFP_${k}/model_ckpt/


python -W ignore CRC_expt.py --save_results_name CRC_results_resnet18_random_pick_${k}.csv \
													    --search_dir ${search_dir} \
													    --query_dir ${query_dir} \
													    --annotated_dir ${annotated_dir} \
													    --annotated_img_dir ${annotated_img_dir} \
													    --test_dir ${test_dir} \
													    --num_img_to_rev ${k} \
				                                        --use_aux_query 1 \
				                                        --use_query_refinement 1 \
				                                        --ret_strategy "random_pick" \
				                                        --save_prediction_csv ${csv_model_save_dir}/random_pick_${k}/ \
				                                        --log_dir ${log_dir}/random_pick_${k}/ \
				                                        --label_pos 2 \
				                                        --label_set ADI  BACK  DEB  LYM  MUC  MUS  NORM  STR  TUM \
				                                        --model_save_path ${csv_model_save_dir}/random_pick_${k}/model_ckpt/				                                       



python -W ignore CRC_expt.py --save_results_name CRC_results_resnet18_front_mid_end_${k}.csv \
													    --search_dir ${search_dir} \
													    --query_dir ${query_dir} \
													    --annotated_dir ${annotated_dir} \
													    --annotated_img_dir ${annotated_img_dir} \
													    --test_dir ${test_dir} \
													    --num_img_to_rev ${k} \
				                                        --use_aux_query 1 \
				                                        --use_query_refinement 1 \
				                                        --ret_strategy "front_mid_end_ret" \
				                                        --save_prediction_csv ${csv_model_save_dir}/front_mid_end_${k}/ \
				                                        --log_dir ${log_dir}/front_mid_end_${k}/ \
				                                        --label_pos 2 \
				                                        --label_set ADI  BACK  DEB  LYM  MUC  MUS  NORM  STR  TUM \
				                                        --model_save_path ${csv_model_save_dir}/front_mid_end_${k}/model_ckpt/


python -W ignore CRC_expt.py --save_results_name CRC_results_resnet18_top_k_${k}.csv \
													    --search_dir ${search_dir} \
													    --query_dir ${query_dir} \
													    --annotated_dir ${annotated_dir} \
													    --annotated_img_dir ${annotated_img_dir} \
													    --test_dir ${test_dir} \
													    --num_img_to_rev ${k} \
				                                        --use_aux_query 1 \
				                                        --use_query_refinement 1 \
				                                        --ret_strategy "top_k_ret" \
				                                        --save_prediction_csv ${csv_model_save_dir}/top_k_ret_${k}/ \
				                                        --log_dir ${log_dir}/top_k_ret_${k}/ \
				                                        --label_pos 2 \
				                                        --label_set ADI  BACK  DEB  LYM  MUC  MUS  NORM  STR  TUM \
				                                        --model_save_path ${csv_model_save_dir}/top_k_ret_${k}/model_ckpt/


