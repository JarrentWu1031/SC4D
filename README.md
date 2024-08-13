# SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer (ECCV 2024)
Zijie Wu<sup>1,2</sup>, Chaohui Yu<sup>2</sup>, Yanqin Jiang<sup>2</sup>, Chenjie Cao<sup>2</sup>, Fan Wang<sup>2</sup>, Xiang Bai<sup>1✉</sup> <br>
<sup>1</sup>Huazhong University of Science and Technology (HUST), <sup>2</sup>DAMO Acadamy, Alibaba Group

[**Project Page**](https://sc4d.github.io/) | [**Paper**](https://arxiv.org/abs/2404.03736) | [**Video Demo**](https://www.youtube.com/watch?v=SkpTEuX4B5c) 

![Demo GIF](https://github.com/JarrentWu1031/SC4D/blob/main/assets/teaser.gif)

<div align=center>
<img src="https://github.com/JarrentWu1031/SC4D/blob/main/assets/teaser.png" width=85%>
</div>


# News
[**2024.08.12**] The training and inference code for **SC4D** is available now! The cleaned code performs comparable or even better than reported in the main paper! Besides, the training time has been reduced to about 40 minutes per example for video-to-4D generation and 30 minutes for motion transfer (all tested on a single Tesla V100 GPU), which is approximately 67% of the main paper! <br>
[**2024.07.04**] **SC4D** has been accepted by [ECCV2024](https://eccv.ecva.net/)! The revised version will be online soon! <br>
[**2024.04.04**] The paper of **SC4D** is available at [Arxiv](https://arxiv.org/abs/2404.03736)! <br>
[**2024.03.14**] The [project page](https://sc4d.github.io/) of **SC4D** is available now! We attach more examples of video-to-4D generation and motion transfer application in the project page than in the main paper! <br>

# Installation
```bash
# it is recommanded to use conda
conda create -n sc4d python=3.9
conda activate sc4d

# install dependencies
pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# a version of cuda knn
git clone https://github.com/unlimblue/KNN_CUDA.git
pip install ./KNN_CUDA
```

# Data preparation
We use dataset from [Consistent4D](https://github.com/yanqinJiang/Consistent4D) as our [training data](https://drive.google.com/file/d/1mJNhFKvzZ-8icAw6KC-W-sf7JmmmMUkx/view?usp=sharing) & [testing data](https://drive.google.com/file/d/1jn18kA2FfKMnyQ6fisIn8rhBI0dr3NFk/view?usp=sharing). 
If you want to use your personal data, please split the video to images and name them as "0.png", "1.png"...
Then run command like:
```bash
python process.py <data to process> --outdir <output dir>

# for example
python process.py ./input_data/dancing_spiderman --outdir ./data/dancing_spiderman
```

# Video-to-4D Generation
**Training.**
Use command like:
```bash
python main_sc4d.py train_dynamic=True input_folder=./data/dancing_spiderman save_path=./logs/dancing_spiderman
```
**Inference.**
Use command like:
```bash
python main_sc4d.py save_path=./logs/dancing_spiderman test_stage=s2 render_type=fixed test_azi=0
```
Please check `./configs/sc4d.yaml` for more options.

# Motion Transfer 
Once Video-to-4D generation is finished, the learned motion can be transferred to specific identity according to text descriptions.

**Training.**
Use command like:
```bash
python main_sc4d_mt.py train_dynamic=True save_path=./logs/dancing_spiderman prompt="A photo of Vegeta"
```
**Inference.**
Use command like:
```bash
python main_sc4d_mt.py save_path=./logs/dancing_spiderman test_stage=s3 render_type=fixed test_azi=0
```
Please check `./configs/sc4d_mt.yaml` for more options.

# Citation
If you find our work useful for your research, please star this repo and cite our paper. Thanks!
```
@article{wu2024sc4d,
    author = {Wu, Zijie and Yu, Chaohui and Jiang, Yanqin and Cao, Chenjie and Wang, Fan and Bai, Xiang.},
    title  = {SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer},
    journal = {arxiv:2404.03736},
    year   = {2024},
```

# Acknowledgement 
Our code is based on [Dreamgaussian](https://github.com/dreamgaussian/dreamgaussian) and [SC-GS](https://github.com/yihua7/SC-GS). We thank the authors for their great works! <br>

