
## Efficient Face Super-Resolution via Wavelet-based Feature Enhancement Network (ACMMM 2024)

[Paper (Arxiv)]() | [Project Page]() 

## Installation and Requirements 
I have trained and tested the codes on
- Ubuntu 20.04
- CUDA 11.1  
- Python 3.8, install required packages by `pip install -r requirements.txt`

## Getting Started
Download Our Pretrain Models and Test Dataset. Additionally, we offer our FSR results in orginal paper.
#### Note：Test results are slightly different from the original paper because the model weights were obtained by re-training after organizing our codes.
- [Pretrain_Models]()  
- [Test_Datasets](https://drive.google.com/file/d/1EW-DZvmIPzMQcYrrwspODoKgFA0oBeR2/view?usp=drive_link)
- [FSR_Results_in_Orginal_Paper](https://drive.google.com/file/d/136DlSB1FvI8timRgDL1WRIx8JyKvtwdO/view?usp=drive_link)

### Test with Pretrained Models

```
# On CelebA Test set
python test.py --gpus 1 --model wfen --name wfen \
    --load_size 128 --dataset_name single --dataroot /path/to/datasets/test_datasets/CelebA1000/LR_x8_up/ \
    --pretrain_model_path ./pretrain_models/wfen/wfen_best.pth \
    --save_as_dir results_celeba/wfen
```

```
# On Helen Test set
python test.py --gpus 1 --model wfen --name wfen \
    --load_size 128 --dataset_name single --dataroot /path/to/datasets/test_datasets/Helen50/LR_x8_up/ \
    --pretrain_model_path ./pretrain_models/wfen/wfen_best.pth \
    --save_as_dir results_helen/wfen
```

### Evaluation
We provide evaluation codes in script `test.sh` for calculate PSNR/SSIM/LPIPS/VIF/Parmas/FLOPs scores.


### Train the Model
The commands used to train the released models are provided in script `train.sh`. Here are some train tips:
- You should download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to train WFEN. Please change the `--dataroot` to the path where your training images are stored.  
- To train WFEN, we simply crop out faces from CelebA without pre-alignment, because for ultra low resolution face SR, it is difficult to pre-align the LR images.  
- Please change the `--name` option for different experiments. Tensorboard records with the same name will be moved to `check_points/log_archive`, and the weight directory will only store weight history of latest experiment with the same name.
- If there's not enough memory, you can turn down the `--batch_size`.
- `--gpus` specify number of GPUs used to train. The script will use GPUs with more available memory first. To specify the GPU index, uncomment the `export CUDA_VISIBLE_DEVICES=`.

```
# Train Code
CUDA_VISIBLE_DEVICES=0,1 python train.py --gpus 2 --name wfen --model wfen \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 128 \
    --dataroot /path/to/datasets/CelebA --dataset_name celeba --batch_size 32 --total_epochs 150 \
    --visual_freq 100 --print_freq 10 --save_latest_freq 500
```

## Acknowledgements
This code is built on [Face-SPARNet](https://github.com/chaofengc/Face-SPARNet). We thank the authors for sharing their codes.

## :e-mail: Contact
If you have any question, please email `lewj2408@gmail.com` or `cswjli@bupt.edu.cn`
