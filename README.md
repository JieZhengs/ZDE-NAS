# A implementation for the paper "Zero-shot Differential Evolution Neural Architecture Search for High-incidence Cancer Prediction"

## Citation
```
@article{xu2025adaptive,
  title={Adaptive Multi-particle Swarm Neural Architecture Search for High-incidence Cancer Prediction},
  author={Xu, Liming and Zheng, Jie and He, Chunlin and Wang, Jing and Zheng, Bochuan and Lv, Jiancheng},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2025},
  publisher={IEEE}
}

@INPROCEEDINGS{FPSO,
  title={A Flexible Variable-length Particle Swarm Optimization Approach to Convolutional Neural Network Architecture Design},
  author={Huang, Junhao and Xue, Bing and Sun, Yanan and Zhang, Mengjie},
  booktitle={2021 IEEE Congress on Evolutionary Computation (CEC)},
  year={2021},
  pages={934-941},
  doi={10.1109/CEC45853.2021.9504716}
}

@ARTICLE{EPCNAS,
  author={Huang, Junhao and Xue, Bing and Sun, Yanan and Zhang, Mengjie and Yen, Gary G.},
  journal={IEEE Transactions on Evolutionary Computation (Early Access)},
  title={Particle Swarm Optimization for Compact Neural Architecture Search for Image Classification},
  year={2022},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TEVC.2022.3217290}}

@ARTICLE{10132401,
  author={Huang, Junhao and Xue, Bing and Sun, Yanan and Zhang, Mengjie and Yen, Gary G.},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  title={Split-Level Evolutionary Neural Architecture Search With Elite Weight Inheritance},
  year={2023},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TNNLS.2023.3269816}}
```

## Requirements

- `python 3.9`
- `Pytorch >= 1.8`
- `torchvison`
- `opencv-python`

## Data

Download lC25000、BreakHis and CRC-5000 datasets, and place them in `global.ini` file.

- LC25000 dataset from [here](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- BreakHis dataset from [here](https://www.kaggle.com/datasets/ambarish/breakhis)
- Colorectal dataset from [here](https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist)
    
    
## Folder Structure
- datasets : store dataset
- load_dataset: load data for train eval and test
- log : store train eval and test log
- populations : store population err flops gbest params pbest population information
- scripts : According to the data set corresponding to the template folder, the corresponding network script is generated for training evaluation
- template: Network files constructed from different datasets
- trained_models : stored model
- compute_zen_score : Zero-shot proxy
- evaluate : Architecture evaluation
- evolve : Weights and particles correspond to the evolution of architecture parameters
- global : Configuration files
- main ：Program main function
- population : Population structure generation


## Training and Testing
# Using ZDE-NAS for Custom Dataset Search and Training

If you want to use this repository to perform search and training on your own dataset, please follow these steps:

## Steps

1. **Store Your Data**  
   Place your data in the `datasets` directory.

2. **Modify Data Loading**  
   Modify the data loading method in the `load_dataset` function to ensure that your data is correctly loaded.

3. **Set Up Configuration File**  
   Configure the settings in the `global.ini` file before running the search and training:

   ```ini
    [settings]
    pop_size = 20
    num_iteration = 20
    diff_rate = 0.4
    crossover_rate = 0.5
    dataset = your data name
    
    [network]
    max_length = 10
    image_channel = 3
    max_output_channel = 128
    num_class = 8
    epoch_test = 300
    Tp = 1.5e6
    Tf = 150e6
    wp = -0.01, -1
    wf = -0.01, -1
   ```

4. **Run the Program.**  
   Execute the main.py file.
   
# Notice
If you would like to receive full source results of ZDE-NAS, please contact the author Jie Zheng at zjpdd0905@stu.cwnu.edu.cn. Please indicate your purpose of use, I will reply the first moment I see the email, thank you!

## Reference codes:
- https://github.com/HuangJunh/SLE-NAS
- https://github.com/HuangJunh/EPCNAS
- https://github.com/HuangJunh/FPSO
- https://github.com/JieZhengs/AMPS-NAS
- https://github.com/JieZhengs/TM-NAS
