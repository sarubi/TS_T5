# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from source.evaluate import simplify_file, evaluate_all_metrics

features_kwargs = {
    'DependencyTreeDepthRatioFeature': {'target_ratio': -1},
    'DependencyTreeLengthRatioFeature': {'target_ratio': -1},
    'DifficultWordsRatioFeature': {'target_ratio': -1},
    'WordCountRatioFeature': {'target_ratio': -1}
}

experiment_name="exp_1713803608466002_epoch_5_model_eval_loss"
dataset_base_dir_name=Path("/nethome/sarubi/A8/ptm_access_based_ft/TS_T5_myedit/LLM_based_control_rewrite/experiments/data_filtered_regression_model/T5_ft/f4_maxdepdepth_maxdeplength_diffwords_wc")

datasets_to_evaluate = {
    "gold_ratios" : "processed_data/wikilarge.val.maxdepdepth_above_6.v2_wo_line_46/gold_ratios",
    "lr_ratios" : "processed_data/wikilarge.val.maxdepdepth_above_6.v2_wo_line_46/0_linear_regression_models",
    "roberta_ratios" : "processed_data/wikilarge.val.maxdepdepth_above_6.v2_wo_line_46/roberta_lr",
    "fixed_ratios" : "processed_data/wikilarge.val.maxdepdepth_above_6.v2_wo_line_46/0_fr_ratios"
}

for dataset_name_key, filepath in datasets_to_evaluate.items():
    output_dir = dataset_base_dir_name / f'{dataset_name_key}-filtered_wiki.maxdepdepth_above_6.v2_wo_line_46.valid-t5ft-{experiment_name}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluate dataset: {filepath}, output will be saved to: {output_dir}")

    # Define the source file paths
    complex_source_filepath = dataset_base_dir_name / f'{filepath}/filtered_wiki.maxdepdepth_above_6.v2_wo_line_46.valid.src'
    simple_target_filepath = dataset_base_dir_name / f'{filepath}/filtered_wiki.maxdepdepth_above_6.v2_wo_line_46.valid.tgt'
    complex_processed_filepath = dataset_base_dir_name / f'{filepath}/filtered_wiki.maxdepdepth_above_6.v2_wo_line_46.valid.src_processed'

    # List of source files and their corresponding output file names
    files_to_copy = [
        (complex_source_filepath, output_dir / 'input.txt'),
        (simple_target_filepath, output_dir / 'gold_ref.txt'),
        (complex_processed_filepath, output_dir / 'input.processed.txt')
    ]

    # Function to copy the first 200 lines from each file
    def copy_first_200_lines(source_path, dest_path):
        with open(source_path, 'r', encoding='utf-8') as source_file:
            with open(dest_path, 'w', encoding='utf-8') as dest_file:
                for _ in range(200):
                    line = source_file.readline()
                    if not line:
                        break
                    dest_file.write(line)

    # Apply the function to each file pair
    for source, dest in files_to_copy:
        copy_first_200_lines(source, dest)

    output_filepath = output_dir / f'output.txt'

    # model_dirname=None will use the last trained model from the folder experiments
    repo_dir = Path(__file__).resolve().parent.parent
    model_dirname = repo_dir / "experiments" / experiment_name
    simplify_file(output_dir / 'input.processed.txt', output_filepath, features_kwargs, model_dirname=model_dirname)

    print(evaluate_all_metrics(output_dir / 'input.txt', output_filepath, [output_dir / 'gold_ref.txt']))

# CUDA_VISIBLE_DEVICES=3 python scripts/maxdepth_above_6_val_on_different_ratios_evaluate.py > experiments/exp_1713803608466002_epoch_5_model_eval_loss/logs_val_maxdepth_above_6_evaluate_across_all_ratios