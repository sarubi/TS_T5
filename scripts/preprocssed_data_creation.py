# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
from source import helper
from functools import lru_cache
from multiprocessing import Pool, Lock
import numpy as np
import shutil
from pathlib import Path
import pandas as pd

from source.resources import DUMPS_DIR, NEWSELA_DATASET, PHASES, WORD_FREQUENCY_FILEPATH, \
    get_data_filepath, PROCESSED_DATA_DIR, \
    DATASETS_DIR, WIKILARGE_DATASET, WORD_EMBEDDINGS_NAME, download_glove
from source.helper import tokenize, yield_lines, load_dump, dump, write_lines, count_line, \
    print_execution_time, save_preprocessor, yield_sentence_pair


def preprocess_dataset(dataset):
    # download_requirements()
    preprocessed_data_dir = PROCESSED_DATA_DIR / dataset
    preprocessed_data_dir.mkdir(parents=True, exist_ok=True)
    print(f'Preprocessing dataset: {dataset}')

    for phase in PHASES:
        # for phase in ["valid", "test"]:
        complex_filepath = get_data_filepath(dataset, phase, 'complex')
        simple_filepath = get_data_filepath(dataset, phase, 'simple')

        complex_output_filepath = preprocessed_data_dir / complex_filepath.name
        simple_output_filepath = preprocessed_data_dir / simple_filepath.name
        if complex_output_filepath.exists() and simple_output_filepath.exists():
            continue

        print(f'Prepocessing files: {complex_filepath.name} {simple_filepath.name}')

        ratios_path = DATASETS_DIR / dataset / f'{dataset}.{phase}.ratio.csv'
        prepend_ratios_to_source_text(complex_filepath, ratios_path, complex_output_filepath)
        # write_lines(processed_complex_sentences_list, complex_output_filepath)
        shutil.copy(simple_filepath, simple_output_filepath)

    print(f'Preprocessing dataset "{dataset}" is finished.')


def prepend_ratios_to_source_text(source_file_path, ratios_file_path, output_file_path):
    # Read the source sentences from the file
    with open(source_file_path, 'r') as file:
        source_sentences = file.readlines()

    # Read the ratio CSV file into a DataFrame
    ratios_df = pd.read_csv(ratios_file_path)

    # Create a new column 'current_line' which starts from 1 up to the length of the DataFrame
    ratios_df['current_line'] = range(1, len(ratios_df) + 1)

    # Open the output file for writing the new source lines
    with open(output_file_path, 'w') as output_file:
        # Iterate over each source sentence and corresponding row in the DataFrame by line number
        for index, source_sentence in enumerate(source_sentences):
            # Adjust index to match 'current_line' which starts from 1
            line_number = index + 1

            # Retrieve the relevant ratios for the current sentence using 'current_line'
            if line_number in ratios_df['current_line'].values:
                row = ratios_df[ratios_df['current_line'] == line_number].iloc[0]
                max_dep_depth_ratio = "{:.2f}".format(float(row['MaxDepDepth_ratio']))
                max_dep_length_ratio = "{:.2f}".format(float(row['MaxDepLength_ratio']))
                diff_words_ratio = "{:.2f}".format(float(row['DiffWords_ratio']))
                word_count_ratio = "{:.2f}".format(float(row['WordCount_ratio']))

                # Create the new source line with ratios prepended
                new_source_line = f"DTD_{max_dep_depth_ratio} DTL_{max_dep_length_ratio} DW_{diff_words_ratio} WC_{word_count_ratio}  {source_sentence}"

                # Write the new source line to the output file
                output_file.write(new_source_line)
            else:
                print(f"Warning: No ratio data available for line {line_number}")


if __name__ == '__main__':
    preprocess_dataset(WIKILARGE_DATASET)