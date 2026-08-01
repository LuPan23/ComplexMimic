<div align="center">

# 🤖 [ECCV 2026] ComplexMimic: Human-Scene Interaction Imitation in Complex 3D Environments


[![arXiv](https://img.shields.io/badge/arXiv-2607.02034-b31b1b.svg)](https://arxiv.org/abs/2607.02034)
[![Paper](https://img.shields.io/badge/Paper-PDF-yellow.svg)](https://arxiv.org/pdf/2607.02034)
[![Python](https://img.shields.io/badge/Python-3.8-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black.svg?logo=github)](https://github.com/LuPan23/ComplexMimic)

</div>


## 🛠️ Dependencies

To create the environment, follow the following instructions: 

1. Create new conda environment and install pytroch:


```
conda create -n complexmimic python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

2. Download and setup [Isaac Gym](https://developer.nvidia.com/isaac-gym). 

3. Install [SMPLSim](https://github.com/ZhengyiLuo/SMPLSim) to automatically create the SMPL humanoid. Please run `pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master` to SMPLSim.

4. Download SMPL paramters from [[link]](https://drive.google.com/drive/folders/1MgO5L89rbhL9n_qI-Wp_oX7boOt2jaxc?usp=sharing) . Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. 

```
|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl
```



## 📦 Datasets

Preprocessed datasets are provided as follows:

- **TRUMANS**: [[link]](https://drive.google.com/drive/folders/1r9MTEYF1X8EDEt6IIzUKIapO1np07dbO?usp=sharing)
- **LINGO**: [[link]](https://drive.google.com/drive/folders/1MDXof_8ItWrgar7D0yaL4ww7LVIn-TBb?usp=sharing)
- **GIMO**: [[link]](https://github.com/y-zheng18/GIMO)

Due to the dataset license, the preprocessed GIMO data cannot be redistributed. Please obtain the original GIMO dataset from the [official repository](https://github.com/y-zheng18/GIMO) and generate the required files using the provided preprocessing scripts.

The data processing scripts are located in:

```text
./data_process
```

The datasets should be organized as follows:

```text
data/
├── TRUMANS/
│   ├── trumans_scene_mesh/
│   ├── trumans_motion_train.pkl
│   ├── trumans_train.json
│   ├── trumans_general_policy_train.json
│   └── trumans_inference.json
│
├── LINGO/
│   ├── lingo_scene_mesh/
│   ├── lingo_motion_inference.pkl
│   └── lingo_inference.json
│
└── GIMO/
    ├── gimo_scene_mesh/
    ├── gimo_motion_inference.pkl
    └── gimo_inference.json
```

## Pretrained models
The pretrained models are available at [[Link]](https://drive.google.com/drive/folders/12C2FBem8PKhzi25s5PAw-i_0O_X5HFum?usp=sharing). You can download the corresponding pretrained models and place them under the `output/HumanoidIm` directory as follows:

```
|-- output
    |-- HumanoidIm
        |-- general_policy
        |-- imitation_expert
        |-- interaction_expert
```

## 🔍 Test
You can directly test the performance of the pre-trained model as follows
1. Test the teacher model

```bash
bash complexmimic/scripts/inference_interaction_expert_trumans.sh
bash complexmimic/scripts/inference_interaction_expert_lingo.sh
bash complexmimic/scripts/inference_interaction_expert_gimo.sh
```

2. Test the student model

```bash
bash complexmimic/scripts/inference_general_policy_trumans.sh
bash complexmimic/scripts/inference_general_policy_lingo.sh
bash complexmimic/scripts/inference_general_policy_gimo.sh
```


## 🚀 Train
1. Download TRUMANS datasets and set the following structure

```
|-- data
    |-- TRUMANS
        |-- trumans_scene_mesh
        |-- trumans_motion_train.pkl
        |-- trumans_motion_scene_mapping_train.json
        |-- trumans_motion_scene_mapping_general_policy_train.json
        |-- trumans_motion_scene_mapping_inference.json
```

2. Train the teacher model

```bash
bash complexmimic/scripts/train_imitation_expert.sh
bash complexmimic/scripts/train_interaction_expert.sh
```

3. Train the student model

```bash
bash complexmimic/scripts/train_general_policy.sh
```

## 📚 References

Our implementation is based on [PHC](https://github.com/ZhengyiLuo/PHC) and [InterMimic](https://github.com/Sirui-Xu/InterMimic). We would like to thank them.

## 📄 Citation


If you find this work useful, please cite our publication:

L. Pan and H. Zhao, “ComplexMimic: Human-Scene Interaction Imitation in Complex 3D Environments,” in *European Conference on Computer Vision (ECCV)*, 2026.

BibTeX:

```bibtex
@inproceedings{pan2026complexmimic,
  title     = {ComplexMimic: Human-Scene Interaction Imitation in Complex 3D Environments},
  author    = {Lu Pan and Hongwei Zhao},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```