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
## Datasets
In the experiments section of our paper, we primarily utilized two datasets: the 3D-OVS dataset and the LERF dataset.

The 3D-OVS dataset is accessible for download via the following link: [Download 3D-OVS Dataset](https://drive.google.com/drive/folders/1kdV14Gu5nZX6WOPbccG7t7obP_aXkOuC?usp=sharing) .

The LERF dataset can be downloaded via the following link: [Download Expanded LERF Dataset](https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view?usp=sharing).

### Usage
+ First, you should set the API_KEY and URL for the large language model in config.yaml.
+ Second,  mkdir data,  then, Put the image in data.Then, execute the following commands:
```
python convert.py -s data/you_data
```

```
python train.py -s data/you_data -m output
```
```
python main.py --config.yaml
```
### S-YOLOv8 will be released publicly after the paper is accepted.
