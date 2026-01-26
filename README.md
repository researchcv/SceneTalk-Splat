# SceneTalk-Splat: Enabling Conversational Interaction with 3D Gaussian Splatting and Large Language Models

### Installation
+ Install `Python >= 3.10`.
+ Install `torch >= 2.4`. We have tested on `torch==2.4.1+cu124`, but other versions should also work fine.
+ Clone our repo
```
https://github.com/Lee-Luc/SceneTalk-Splat.git --recursive
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
### I'm very sorry, but due to the large file size of the S-YOLOv8 model, it cannot be submitted to the repository for the time being. A download link will be provided later.Therefore, to make our method available to readers, we used the existing YOLOv8 instead. This is only temporary, and we are finding ways to submit our own S-YOLOv8 model.
