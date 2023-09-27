# PixelRec: An Image Dataset for Benchmarking Recommender Systems with Raw Pixels



# Dataset



### Interaction Data

Interactions can be downloaded from https://sandbox.zenodo.org/record/1166723, containing **Pixel200K**, **Pixel1M**, **Pixel8M** and **PixelRec**.  The item popularity distributions are shown as follows : 

<div align=center><img src="https://github.com/website-pixelrec/PixelRec/blob/main/dataset/statistics/item_rank.png"/></div>


For more detailed statistics of interactions, see  `dataset/statistics`.

**We currently provide a sampled version including 50,000 users. The complete dataset will be made publicly available after the paper is accepted. The sampled dataset can be obtained through:**

- zenodo : https://sandbox.zenodo.org/record/1241482

- google drive:  

  https://drive.google.com/drive/folders/1bQPgM-6yAnzcD0jKBoUUheA9LL5xnCHG?usp=sharing 

  &  https://drive.google.com/file/d/1SZ7hj1HDVCV_WDidSOFGgCVhx1rpWE31/view?usp=sharing  

  Due to the file size upload limit, we have evenly divided the image files into two folders, i.e. cover1 and cover2 in the link. 



### Item Information Data

See  `dataset/item_info.csv`.
The item_info.csv contains item features in Pixel200K, the column names of the file are as follows:
| item_id | view_number | comment_number | thumbup_number | share_number | coin_number | favorite_number | barrage_number | title_en |   title | description |  tag |
| ------: | ----------: | -------------: | -------------: | -----------: | ----------: | --------------: | -------------: | ---: | ----------: | --------: | --------: |

The entire item information data will be available once paper is accepted. We will provide the text features (description and tag) in English in the future.



# Experiments

## Environments

```
Pytorch==1.10.2
cudatoolkit==11.2.1
python==3.9.7
```

See requirements.txt for other packages:
```python
pip install -r requirements.txt
```


## Run Baselines

To run the baselines:

- Download the interaction data and images.

- Generate lmdb database from the images:

```python
cd code && python generate_lmdb.py
```

- You can choose different `yaml` files to run different baselines, the `yaml` files are under folders `IDNet`, `PixelNet` ,`ViNet` and `overall`

To run IDNet, for example, run `SASRec` model on one card:

```python
python main.py --device 0 --config_file IDNet/sasrec.yaml overall/ID.yaml
```

Change the `IDNet/sasrec.yaml` to run other IDNet baselines.



To run PixelNet, for example, run `SASRec` model with `ViT` encoders on four cards:

```python
python main.py --device 0,1,2,3 --config_file PixelNet/sasrec.yaml overall/ViT.yaml
```

Change  `PixelNet/sasrec.yaml` to run other PixelNet baselines with `ViT` as item encoder,  change  `overall/ViT.yaml` to run `sasrec` model with other image encoders.



To run ViNet, e.g. run `VBPR` model on one card:

```python
python main.py --device 0 --config_file ViNet/vbpr.yaml
```

Change  `ViNet/vbpr.yaml` to run other ViNet



Note: you may need to modify some path in files under folders `ViNet` and `overall` and file `generate_lmdb.py` , depending on where you put the downloaded data.



# Hyper Parameters

> Hyper parameter range : 
>
> embedding size [128, 512, 1024, 2048, 4096, 8192]
>
> learning rate [0.000001, 0.00005, ... , 0.001]
>
> weight decay [0, 0.01, 0.1]
>
> batch size [64, 128, 256, 512, 1024]



**Hyper-parameter details of IDNet. $\gamma$,  $\beta$ and $B$ are the learning rate, weight decay and batch size respectively.**

| Method (IDNet) | Model Parameters                                             | Training Parameters                                          |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MF     | dropout prob [0]    embedding size [4096]                    | γ [0.0001]   B [512]     β [0]                               |
| FM        | embedding size [4096]                                        | γ [0.00005]   B [64]     β [0]                               |
| DSSM      | dnn layer number [0]    embedding size [4096]                | γ [0.0001]   B [64]     β [0]                                |
| LightGCN  | step [2]    embedding size [256]                             | γ [0.0005]   B [1024]     β [0.01]                           |
| SASRec    | trm layer number [2]    inner size [2]    embedding size [512] | γ [0.00005]   B [64]     β [0.1]                             |
| BERT4Rec  | mask ratio [0.6]    trm layer number [2]    inner size  [1]    embedding size [512] | γ [0.00005]   B [64]     β [0.1]                             |
| LightSANs  | k [3]    trm layer number [1]  embedding size [512] | γ [0.00005]   B [512]     β [0.1]                             |
| GRU4Rec   | dropout prob [0]    gru layer number [1]    inner size  [2]    embedding size [2048] | γ [0.0001]   B [64]     β [0.01]                             |
| NextItNet | block number [3]    embedding size [1024]                    | γ [0.0005]   B [64]     β [0.01]                             |
| SRGNN     | step [2]    embedding size [512]                             | γ [0.00005]   B [64]     β [0.01]                            |
| VisRank   | visual feature [RN_2048]    method [maximum]                 |                                                              |
| VBPR      |                                                              | id γ [0.001]    id β [0]    visual γ [0.0001]    visual β [0.1] |
| ACF       | embedding size [128]                                         | γ [0.0001]   B [64]     β [0.1]                              |



**For the most architectures, PixelNet uses the same hyperparameters as its IDNet, with a few exceptions here. The embedding size refers to the hidden dimension of the user encoder.**

| Method  (PixelNet) | Model Parameters | Training Parameters |
| ------ | ---------------- | ------------------- |
| SASRec    | trm layer number [2]    inner size [2]    embedding size [512] | γ [0.0001]   B [64]     β [0.1]                             |
| BERT4Rec  | mask ratio [0.6]    trm layer number [2]    inner size  [1]    embedding size [512] | γ [0.0001]   B [64]     β [0.1]                             |
| LightSANs  | k [3]    trm layer number [1]  embedding size [512] | γ [0.0001]   B [512]     β [0.1]                             |
| NextItNet | block number [3]    embedding size [1024]                    | γ [0.0001]   B [64]     β [0.01]                             |
| SRGNN     | step [2]    embedding size [512]                             | γ [0.0001]   B [512]     β [0.01]                          |



**In PixelNet, we adopt different learning rate and weight decay between the image encoder and the rest of the model structures. Here are the hyper-parameter for tuning the image encoders.**

| Image Encoder                            | Hyper Parameter       |
| ---------------------------------------- | --------------------- |
| RN50, RN50x4, RN50x16, RN50x64, ResNet50 | γ [0.0001]   β [0.01] |
| ViT, Swin-T, Swin-B, BEiT                | γ [0.0001]   β [0]    |





If our work has been of assistance to your work, please cite our paper as :  

```
@article{cheng2023image,
  title={An Image Dataset for Benchmarking Recommender Systems with Raw Pixels},
  author={Cheng, Yu and Pan, Yunzhu and Zhang, Jiaqi and Ni, Yongxin and Sun, Aixin and Yuan, Fajie},
  journal={arXiv preprint arXiv:2309.06789},
  year={2023}
}
```




#### The laboratory is hiring research assistants, interns, doctoral students, and postdoctoral researchers. Please contact the corresponding author for details.
