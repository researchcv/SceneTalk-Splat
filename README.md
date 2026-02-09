# Enhancing 3D Scene Understanding through Conversational Interaction with Gaussian Splatting and Language Models

### Installation
+ Install `Python >= 3.10`.
+ Install `torch >= 2.4`. We have tested on `torch==2.4.1+cu124`, but other versions should also work fine.
+ Clone our repo
```
https://github.com/researchcv/SceneTalk-Splat.git --recursive
```
+ Install dependencies:
```
pip install plyfile tqdm opencv-python-headless joblib
pip install -r requirements.txt
```
+ Install submodules:
```
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
### Usage
+ First, you need to select the API_KEY and URL of the large language model you like in the configuration file
+ Second,  mkdir data,  then, Put the image in data
```
python convert.py -s data/you_data
```

```
python train.py -s data/you_data -m output
```
```
python main.py --config.yaml
```
