#!/bin/bash
#SBATCH -A ashishmenon
#SBATCH -n 12
#SBATCH --nodelist=gnode85
#SBATCH --wait-all-nodes=0
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END



module load cuda/10.0
module load cudnn/7.3-cuda-10.0

module load cuda/10.0
module load cudnn/7.3-cuda-10.0


CODE_path='/home/ashishmenon/SMILY++/pyfiles/Metric_learning_Interactive_ret_CRC/acpr_codebase'
DOWNLOAD_path='/ssd_scratch/cvit/ashishmenon'

# cd ${DOWNLOAD_path}
# wget -O ICIAR2018_BACH_Challenge.zip https://zenodo.org/record/3632035/files/ICIAR2018_BACH_Challenge.zip?download=1
# unzip ICIAR2018_BACH_Challenge.zip 




cd ${CODE_path}/


python patch_extraction_ICIAR.py  --dataroot ${DOWNLOAD_path}/ICIAR2018_BACH_Challenge/WSI \
						  		  --save_path ${DOWNLOAD_path}/ICIAR2018_BACH_Challenge_saved_patches_labelwise/


mkdir -p ${DOWNLOAD_path}/ICIAR2018_BACH_Challenge_saved_patches_labelwise_modified/Tumor/
mkdir -p ${DOWNLOAD_path}/ICIAR2018_BACH_Challenge_saved_patches_labelwise_modified/Normal/

for f in Benign Benign_Invasive Benign_InSitu Benign_InSitu_Invasive InSitu InSitu_Invasive Invasive
	do
		rsync -aP ${DOWNLOAD_path}/ICIAR2018_BACH_Challenge_saved_patches_labelwise/${f}/ ${DOWNLOAD_path}/ICIAR2018_BACH_Challenge_saved_patches_labelwise_modified/Tumor/
	done

rsync -aP ${DOWNLOAD_path}/ICIAR2018_BACH_Challenge_saved_patches_labelwise/Normal/ ${DOWNLOAD_path}/ICIAR2018_BACH_Challenge_saved_patches_labelwise_modified/Normal/


k=5
class=Tumor
for slide in A01 A02 A03 A04 A05 A06 A07 A09 A10 
	do
		mkdir -p ${DOWNLOAD_path}/ICIAR2018_BACH_Challenge_slide_wise_patches/${slide}/
		rsync -aPq ${DOWNLOAD_path}/ICIAR2018_BACH_Challenge_saved_patches_labelwise_modified/Normal/${slide}/ ${DOWNLOAD_path}/ICIAR2018_BACH_Challenge_slide_wise_patches/${slide}/Normal/
		rsync -aPq ${DOWNLOAD_path}/ICIAR2018_BACH_Challenge_saved_patches_labelwise_modified/Tumor/${slide}/ ${DOWNLOAD_path}/ICIAR2018_BACH_Challenge_slide_wise_patches/${slide}/Tumor/
		
		python save_conv_features2.py --dataroot ${DOWNLOAD_path}/ICIAR2018_BACH_Challenge_slide_wise_patches/${slide} \
								--save_dir ${DOWNLOAD_path}/ICIAR_${slide}_resnet_pretrained_features/ \
								--label_pos 2 \
								--use_fc 0 \
								--use_resnet18 1 \
								--use_resnet34 0 \
								--use_texture_encoder 0 \
								--dataset ICIAR \
								--label_set Normal Tumor
		
		
		rm -rf ${DOWNLOAD_path}/ICIAR_${slide}_resnet_pretrained_features_modified
		python -W ignore ICIAR_expt.py.py           --save_results_name ICIAR_results_resnet18_${slide}_${class}_hybrid_${k}.csv \
												    --search_dir ${DOWNLOAD_path}/ICIAR_${slide}_resnet_pretrained_features \
												    --make_query_test_set 1 \
												    --num_img_to_rev ${k} \
			                                        --use_aux_query 1 \
												    --use_classifier_pred 1 \
			                                        --classes_to_query ${class} \
			                                        --use_query_refinement 1 \
			                                        --ret_strategy "front_mid_end_ret" \
			                                        --save_prediction_csv ${DOWNLOAD_path}/ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/hybrid_${k}_hardest_triplet/ \
			                                        --log_dir ${DOWNLOAD_path}/tensorboard_ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/hybrid_${k}_hardest_triplet/ \
			                                        --label_pos 2 \
			                                        --classifier_pred_top_mid_end 1\
			                                        --label_set Normal Tumor \
			                                        --model_save_path ${DOWNLOAD_path}/ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/hybrid_${k}_hardest_triplet/model_ckpt/ 	                                       

		rm -rf ${DOWNLOAD_path}/ICIAR_${slide}_resnet_pretrained_features_modified
		python -W ignore ICIAR_expt.py.py --save_results_name CAM16_results_resnet18_${slide}_${class}_CNFP_${k}.csv \
												    --search_dir ${DOWNLOAD_path}/ICIAR_${slide}_resnet_pretrained_features \
												    --make_query_test_set 1 \
												    --num_img_to_rev ${k} \
			                                        --use_aux_query 1 \
			                                        --use_classifier_pred 1 \
			                                        --classes_to_query ${class} \
			                                        --use_query_refinement 1 \
			                                        --ret_strategy "front_mid_end_ret" \
			                                        --save_prediction_csv ${DOWNLOAD_path}/ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/CNFP_${k}/ \
			                                        --log_dir ${DOWNLOAD_path}/tensorboard_ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/CNFP_${k}/ \
			                                        --label_pos 2 \
			                                        --only_classifier_pred 1\
			                                        --label_set Normal Tumor \
			                                        --model_save_path ${DOWNLOAD_path}/ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/CNFP_${k}/model_ckpt/ 

		rm -rf ${DOWNLOAD_path}/ICIAR_${slide}_resnet_pretrained_features_modified
		python -W ignore ICIAR_expt.py.py --save_results_name ICIAR_results_resnet18_${slide}_${class}_entropy_based_${k}.csv \
												    --search_dir ${DOWNLOAD_path}/ICIAR_${slide}_resnet_pretrained_features \
												    --make_query_test_set 1 \
												    --num_img_to_rev ${k} \
			                                        --use_aux_query 1 \
			                                        --use_classifier_pred 1 \
			                                        --classes_to_query ${class} \
			                                        --use_query_refinement 1 \
			                                        --ret_strategy "front_mid_end_ret" \
			                                        --save_prediction_csv ${DOWNLOAD_path}/ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/entropy_based_${k}/ \
			                                        --log_dir ${DOWNLOAD_path}/tensorboard_ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/entropy_based_${k}/ \
			                                        --label_pos 2 \
			                                        --entropy_based 1\
			                                        --label_set Normal Tumor \
			                                        --model_save_path ${DOWNLOAD_path}/ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/entropy_based_${k}/model_ckpt/


		rm -rf ${DOWNLOAD_path}/ICIAR_${slide}_resnet_pretrained_features_modified
		python -W ignore ICIAR_expt.py.py --save_results_name ICIAR_results_resnet18_${slide}_${class}_random_${k}.csv \
												    --search_dir ${DOWNLOAD_path}/ICIAR_${slide}_resnet_pretrained_features \
												    --make_query_test_set 1 \
												    --num_img_to_rev ${k} \
			                                        --use_aux_query 1 \
			                                        --use_classifier_pred 0 \
			                                        --classes_to_query ${class} \
			                                        --use_query_refinement 1 \
			                                        --ret_strategy "random_pick" \
			                                        --save_prediction_csv ${DOWNLOAD_path}/ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/random_${k}/ \
			                                        --log_dir ${DOWNLOAD_path}/tensorboard_ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/random_${k}/ \
			                                        --label_pos 2 \
			                                        --label_set Normal Tumor \
			                                        --model_save_path ${DOWNLOAD_path}/ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/random_${k}/model_ckpt/


        rm -rf ${DOWNLOAD_path}/ICIAR_${slide}_resnet_pretrained_features_modified
		python -W ignore ICIAR_expt.py.py --save_results_name ICIAR_results_resnet18_${slide}_${class}_front_mid_end_${k}.csv \
												    --search_dir ${DOWNLOAD_path}/ICIAR_${slide}_resnet_pretrained_features \
												    --make_query_test_set 1 \
												    --num_img_to_rev ${k} \
			                                        --use_aux_query 1 \
			                                        --use_classifier_pred 1 \
			                                        --classes_to_query ${class} \
			                                        --use_query_refinement 1 \
			                                        --ret_strategy "front_mid_end_ret" \
			                                        --save_prediction_csv ${DOWNLOAD_path}/ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/front_mid_end_${k}/ \
			                                        --log_dir ${DOWNLOAD_path}/tensorboard_ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/front_mid_end_${k}/ \
			                                        --label_pos 2 \
			                                        --label_set Normal Tumor \
			                                        --model_save_path ${DOWNLOAD_path}/ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/front_mid_end_${k}/model_ckpt/

        rm -rf ${DOWNLOAD_path}/ICIAR_${slide}_resnet_pretrained_features_modified
		python -W ignore ICIAR_expt.py.py --save_results_name ICIAR_results_resnet18_${slide}_${class}_top_${k}.csv \
												    --search_dir ${DOWNLOAD_path}/ICIAR_${slide}_resnet_pretrained_features \
												    --make_query_test_set 1 \
												    --num_img_to_rev ${k} \
			                                        --use_aux_query 1 \
			                                        --use_classifier_pred 1 \
			                                        --classes_to_query ${class} \
			                                        --use_query_refinement 1 \
			                                        --ret_strategy "top_k_ret" \
			                                        --save_prediction_csv ${DOWNLOAD_path}/ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/top_${k}/ \
			                                        --log_dir ${DOWNLOAD_path}/tensorboard_ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/top_${k}/ \
			                                        --label_pos 2 \
			                                        --label_set Normal Tumor \
			                                        --model_save_path ${DOWNLOAD_path}/ICIAR_preds_resnet18_slidewise_annot/${slide}/${class}/top_${k}/model_ckpt/


	done



