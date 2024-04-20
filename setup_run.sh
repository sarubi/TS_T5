
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