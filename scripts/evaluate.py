# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from source.evaluate import evaluate_on_TurkCorpus, evaluate_on_asset, evaluate_on


features_kwargs = {
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.1},
    'DependencyTreeLengthRatioFeature': {'target_ratio': 0.1},
    'DifficultWordsRatioFeature': {'target_ratio': 0.1},
    'WordCountRatioFeature': {'target_ratio': 0.1}
    # # 'WordRatioFeature': {'target_ratio': 1.05},
    # 'CharRatioFeature': {'target_ratio': 0.95},
    # 'LevenshteinRatioFeature': {'target_ratio': 0.75},
    # 'WordRankRatioFeature': {'target_ratio': 0.75},
    # 'DependencyTreeDepthRatioFeature': {'target_ratio': 0.75}
}
evaluate_on_TurkCorpus(features_kwargs, 'test')
evaluate_on_asset(features_kwargs, 'test')

