# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from source.resources import DATASETS_DIR, REPO_DIR, PROCESSED_DATA_DIR
from source.evaluate import simplify_file, evaluate_all_metrics

features_kwargs = {
    'DependencyTreeDepthRatioFeature': {'target_ratio': -1},
    'DependencyTreeLengthRatioFeature': {'target_ratio': -1},
    'DifficultWordsRatioFeature': {'target_ratio': -1},
    'WordCountRatioFeature': {'target_ratio': -1}

    # 'CharRatioFeature': {'target_ratio': 0.95},
    # 'LevenshteinRatioFeature': {'target_ratio': 0.75},
    # 'WordRankRatioFeature': {'target_ratio': 0.75},
    # 'DependencyTreeDepthRatioFeature': {'target_ratio': 0.75}
}

repo_dir = Path(__file__).resolve().parent.parent
model_dirname = repo_dir / "experiments/exp_1713803110061108"

# # *************************** valid dataset ************************************
# output_dir = model_dirname / f'outputs/wikilarge_valid_200/gold_ratio/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1'
# output_dir.mkdir(parents=True, exist_ok=True)
#
# # Define the source file paths
# wikilarge_valid_complex_processed_filepath = PROCESSED_DATA_DIR / 'wikilarge/wikilarge.valid.complex'
#
# # List of source files and their corresponding output file names
# files_to_copy = [
#     (DATASETS_DIR / 'wikilarge/wikilarge.valid.complex', output_dir / 'input.txt'),
#     (DATASETS_DIR / 'wikilarge/wikilarge.valid.simple', output_dir / 'gold_ref.txt'),
#     (wikilarge_valid_complex_processed_filepath, output_dir / 'input.processed.txt')
# ]
#
# # Function to copy the first 200 lines from each file
# def copy_first_200_lines(source_path, dest_path):
#     with open(source_path, 'r', encoding='utf-8') as source_file:
#         with open(dest_path, 'w', encoding='utf-8') as dest_file:
#             for _ in range(200):
#                 line = source_file.readline()
#                 if not line:
#                     break
#                 dest_file.write(line)
#
# # Apply the function to each file pair
# for source, dest in files_to_copy:
#     copy_first_200_lines(source, dest)
#
# output_filepath = output_dir / f'output.txt'
#
# # model_dirname=None will use the last trained model from the folder experiments
# simplify_file(output_dir / 'input.processed.txt', output_filepath, features_kwargs, model_dirname=model_dirname)
#
# print(evaluate_all_metrics(output_dir / 'input.txt', output_filepath, [output_dir / 'gold_ref.txt']))



# # *************************** test maxdep above 6 ************************************
output_dir = model_dirname / f'outputs/wikilarge_test_200_ie_maxdepthabove_6/gold_ratio/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1'
output_dir.mkdir(parents=True, exist_ok=True)

# Define the source file paths
wikilarge_valid_complex_processed_filepath = PROCESSED_DATA_DIR / 'wikilarge/wikilarge.test.complex'

# List of source files and their corresponding output file names
files_to_copy = [
    (DATASETS_DIR / 'wikilarge/wikilarge.test.complex', output_dir / 'input.txt'),
    (DATASETS_DIR / 'wikilarge/wikilarge.test.simple', output_dir / 'gold_ref.txt'),
    (wikilarge_valid_complex_processed_filepath, output_dir / 'input.processed.txt')
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
simplify_file(output_dir / 'input.processed.txt', output_filepath, features_kwargs, model_dirname=model_dirname)

print(evaluate_all_metrics(output_dir / 'input.txt', output_filepath, [output_dir / 'gold_ref.txt']))

