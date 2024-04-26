
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#
#chmod u+x Miniconda3-latest-Linux-x86_64.sh
#
#./Miniconda3-latest-Linux-x86_64.sh

conda create --name ts_t5_ft python=3.7.5

conda activate ts_t5_ft

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html pytorch-lightning==1.6.5


git clone https://github.com/feralvam/easse.git
cd easse
pip install -e .
cd ../


cd TS_T5
pip install -r requirements_by_sarubi.txt
python -m nltk.downloader stopwords
python -m spacy download en_core_web_md
#pip install -r requirements.txt


CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py
CUDA_VISIBLE_DEVICES=3 python scripts/train.py

#CUDA_VISIBLE_DEVICES=3 python scripts/train.py > experiments/logs_epoch_5_batch16_with_gold_ratio_sari_eval
#CUDA_VISIBLE_DEVICES=1 python scripts/train.py > experiments/logs_epoch_1_batch16_with_gold_ratio_sari_eval
#
#CUDA_VISIBLE_DEVICES=1 python scripts/train.py > experiments/logs_epoch_1_batch16_with_gold_ratio_sari_eval_run_2
#
#CUDA_VISIBLE_DEVICES=1 python scripts/generate.py > experiments/exp_1713749698488586/logs_generate
#
#
## val loss default entropy
#CUDA_VISIBLE_DEVICES=1 python scripts/train.py > experiments/logs_epoch_1_batch16_with_model_eval_loss
#
#CUDA_VISIBLE_DEVICES=3 python scripts/train.py > experiments/logs_epoch_5_batch16_with_model_eval_loss
#
#
#CUDA_VISIBLE_DEVICES=1 python scripts/generate.py > experiments/exp_1713803110061108/logs_generate_valid
#CUDA_VISIBLE_DEVICES=1 python scripts/generate.py > experiments/exp_1713803110061108/logs_generate_test_maxdepth_above_6
#
#
#
## sari loss default fixed ratio 0.8
#CUDA_VISIBLE_DEVICES=0 python scripts/train.py > experiments/logs_epoch_1_batch16_with_sari_loss_fr_0.8
#
#CUDA_VISIBLE_DEVICES=3 python scripts/train.py > experiments/logs_epoch_1_correct_batch16_with_sari_loss_fr_0.8_gpu3
#
#CUDA_VISIBLE_DEVICES=4 python scripts/train.py > experiments/logs_debug_default_sari_loss_with_fr_0.8



# train v3 - model eval loss
CUDA_VISIBLE_DEVICES=3 python scripts/train.py > experiments/logs_epoch_5_batch16_with_train_v3_model_eval_loss

CUDA_VISIBLE_DEVICES=1 python scripts/train.py > experiments/logs_epoch_1_batch16_with_train_v3_model_eval_loss

#CUDA_VISIBLE_DEVICES=0 python scripts/train.py > experiments/logs_debug_update_20_with_train_v3_model_eval_loss

