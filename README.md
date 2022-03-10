# Freq-Aware-Depth

This is the PyTorch implementation for `Frequency-Aware Self-Supervised Depth Estimation`.

Our methods are highly generalizable, see what changes are made to the baseline ([Monodepth2](https://github.com/nianticlabs/monodepth2.git)) via referring to `git diff` or `git log` (*Not available in this anonymous submission, because it provides information to identify the authors*), and easily integrate them into your model.


## Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
conda install pytorch=1.7.1 torchvision=0.8.2 -c pytorch 
pip install tensorboardX==1.5     # 1.4 also ok
conda install opencv=3.4.2    # just needed for evaluation, 3.3.1 also ok
```


## Training

By default, models and tensorboard event files are saved to `~/tmp/{model_name}`.
This can be changed with the `--log_dir` flag.

**Train without our contributions:**
```shell
python train.py --model_name {name_you_expect} --disable_auto_blur --disable_ambiguity_mask
```

**Train our full model:**
```shell
python train.py --model_name {name_you_expect}
```

Our methods introduce no more than 10% extra training time and no extra inference time at all.

## KITTI training data

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```
**Warning:** it weighs about **175GB**.


## Predict depth for a single image

```shell
python test_simple.py --image_path assets/test_image.jpg --model_path {pretrained_model_path}
```

## Pretrained Models

*Pretrained models are to ready be released if the paper is accepted, because the links 🔗 provide information to identify the authors, which cannot appear in this anonymous submission.*


## KITTI evaluation

Run
```shell
python export_gt_depth.py --data_path {KITTI_path} --split eigen
```
to export the ground truth depth first.

Then, run
```shell
python evaluate_depth.py --eval_mono --load_weights_folder {model_path}
```
to evaluate the model in `model_path`.

