plot:
	python tools/domain_shift_vis.py \
		--train_normal_dir data/mvtec_ad_2/can/train/good \
		--test_normal_dir data/mvtec_ad_2/can/test_public/good \
		--test_abnormal_dir data/mvtec_ad_2/can/test_public/bad \
		--output_dir ./domain_shift_out_can

PLOT_SHIFT_CLASSES ?= can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts
PLOT_SHIFT_TYPES ?= bad good
PLOT_SHIFT_SPLIT ?= test_public
PLOT_SHIFT_OUT ?= ./domain_shift_idx_plots_sliding

plot_shift:
	for cls in $(PLOT_SHIFT_CLASSES); do \
		for typ in $(PLOT_SHIFT_TYPES); do \
			out_dir=$(PLOT_SHIFT_OUT)/$$cls-$$typ; \
			python tools/plot_idx_domain_variation.py \
				--image_dir data/mvtec_ad_2/$$cls/$(PLOT_SHIFT_SPLIT)/$$typ \
				--output_dir $$out_dir; \
		done; \
	done

plot_slide:
	for cls in $(PLOT_SHIFT_CLASSES); do \
		for typ in $(PLOT_SHIFT_TYPES); do \
			out_dir=$(PLOT_SHIFT_OUT)/$$cls-$$typ; \
			python tools/plot_idx_domain_variation_sliding.py \
				--image_dir data/mvtec_ad_2/$$cls/$(PLOT_SHIFT_SPLIT)/$$typ \
				--output_dir $$out_dir; \
		done; \
	done

DATA_ROOT := ./data/mvtec_ad_2
EXPNAME := dinov2_base_resize_448_448_without_center_crop_rm_patch
ENCODER_NAME := dinov2reg_vit_base_14
DEVICE ?= cuda:0  # 改成 cpu 就跑 CPU

# 如果保持原本 resize 流程，你可以這種寫法，加上要不要 center_crop。
# python dinomaly_mvtec_sep_mvtecad2.py --input_mode resize [--no_center_crop] 
# --slide_train_patches_per_image: 每張圖最多隨機保留 N 個 patch
# --slide_train_patch_ratio: 每張圖保留指定比例的 patch（向下取整，至少 1 個）。只有在沒有設 patches_per_image 時才會用到。

train_and_eval: train_mvtecad2 MVTecAD2_evaluate_test_public

train_mvtecad2:
	python dinomaly_mvtec_sep_mvtecad2.py \
		--data_path $(DATA_ROOT) \
		--encoder_name $(ENCODER_NAME) \
		--save_dir ./saved_results \
		--no_center_crop \
		--input_mode slide --slide_window_size 448 --slide_window_overlap 0.1 --slide_train_patches_per_image 1 \
		--device $(DEVICE) \
		--save_name $(EXPNAME)

MVTecAD2_evaluate_test_public:
	@for obj in $(PLOT_SHIFT_CLASSES); do \
		echo "=== Evaluating test_public $$obj ==="; \
		python mvtec_ad_evaluation/evaluate_experiment.py \
			--anomaly_maps_dir ./saved_results/$(EXPNAME)/anomaly_images_test_public \
			--dataset_base_dir ./$(DATA_ROOT) \
			--evaluated_objects $$obj \
			--output_dir ./saved_results/$(EXPNAME)/dinomaly_evaluate/$$obj; \
	done

inference_mvtecad2:
	python dinomaly_mvtec_sep_mvtecad2_infer.py \
		--data_path ./data/mvtec_ad_2 \
		--save_dir ./saved_results \
		--save_name $(EXPNAME) \
		--no_center_crop \
		--encoder_name $(ENCODER_NAME) \
		--input_mode slide --slide_window_size 448 --slide_window_overlap 0.2 \
		--device $(DEVICE) \
		--items can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts

