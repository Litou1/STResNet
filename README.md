# STResNet
Spatial-Temporal ResNet code for efficient CT image denoising.
This repository is the official implementation of [Efficient and Accurate Spatial-Temporal Denoising Network for LDCT scans](url)


## Requirements
* CUDA > 10.2, cuDNN > 7.6
* Python virualenv
  To install requirements:
  ```setup
  pip install -r requirements.txt
  ```
* Pull docker container 
  ```
  docker pull nvcr.io/nvidia/pytorch:19.11-py3
  ```
<!-- > 📋Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc... -->


<!-- > 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters. -->

## Evaluation
To evaluate the model in test set, run:
```eval
python test.py -opt options/test/<config_name>.json
```
Note that our UCLA data are in h5, set `data_type` to `h5` in json.  
To evaluate with your own dicom images, chage `data_type` to `dicom` in json. You need to prepare your own input folder with each case folder containing all .dcm images. The resulting outputs are saved at result folder.

## Models
Pretrained models are stored in experiemnts\<run_name>\models\xxx.pth folder

## Results

Our model achieves the following performance on image metrics and speed-ups. 

|      |                      | ↑PSNR(dB)           | ↑SSIM           | ↓LPIPS            | Inference time (sec) | Training time  per iter (sec) | Inference Speed-up | Training Speed-up |
|------|----------------------|---------------------|-----------------|:-----------------:|----------------------|-------------------------------|--------------------|-------------------|
| FP32 | SRResNet  (baseline) | 31.31±0.30          | 0.7216±0.0113   | 0.3635±0.0074     | 27.4(446.7*)         | 6.5                           | N/A                | N/A               |
|      | STResNet             | 31.91±0.44          | 0.7265±0.0110   | 0.3715±0.0075     | 14.4(267.0*)         | 3.9                           | 1.67               | 1.65              |
| FP16 | SRResNet             | 32.39±0.52          | 0.7277±0.0111   | 0.3640±0.0075     | 13.8                 | 4.9                           | N/A                | 1.31              |
|      | STResNet             | 32.60±0.64          | 0.7259±0.0111   | 0.3732±0.0076     | 17.0                 | 3.2                           | N/A                | 2.04              |
| INT8 | SRResNet             | 31.15±0.28          | 0.7064±0.0109   | 0.3501±0.0075     | 108.7*               | N/A                           | 4.11               | N/A               |
|      | STResNet             | 31.11±0.30          | 0.7135±0.0109   | 0.3555±0.0076     | 62.8*                | N/A                           | 7.11               | N/A               |

<!-- >  📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 
