# MMP++: Motion Manifold Primitives with Parametric Curve Models

The official repository for <MMP++: Motion Manifold Primitives with Parametric Curve Models> (Lee, T-RO 2024).

> This paper proposes Motion Manifold Primitives++ (MMP++), which can encode and generate a manifold of trajectories, enabling the efficient generation of high-dimensional trajectories, modulation of latent values and viapoints, and online adaptation in the presence of dynamic obstacles.

- *[Paper](https://ieeexplore.ieee.org/document/10637485)*

## Run

### __0. Preparation__

* Edit the [environment.yml](environment.yml) file with the proper PyTorch version (depedning on your CUDA Version), and run the following command:
  ```
  conda create python=3.8 -n MMPpp
  conda activate MMPpp
  pip install -r requirements.txt --progress-bar on
  ```

* Download the pretrained models from [GOOGLE DRIVE](https://drive.google.com/file/d/1qnsqojNl-OfDNPRSKWiNy_vlEHkgn7tQ/view?usp=sharing) and place the "results" directory in the root folder. If you have downloaded the models, you can skip running [train.py](train.py).

### __1. 2D Toy Experiments__

* To train MMP++:

  ```
  python train.py --base_config configs/Toy/Exp2/base_config.yml --config configs/Toy/Exp2/mmppp.yml --model.z_dim 2 --run mmppp_zdim2 --device 0
  ```
* To train IMMP++:

  ```
  python train.py --base_config configs/Toy/Exp2/base_config.yml --config configs/Toy/Exp2/immppp.yml --model.z_dim 2 --model.iso_reg 10 --run immppp_zdim2_reg10 --device 0
  ```
* Trained models are saved to [results/toy/exp1](results/toy/exp1/).
* To see the trained results, open the [IPython Notebook](1-2D_toy_results_analysis.ipynb).

### __2. 7-DoF Robot Arm (Manifold Example)__

* To train MMP++:

  ```
  python train.py --base_config configs/Robot/robot_manifold/base_config.yml --config configs/Robot/robot_manifold/mmppp.yml --model.z_dim 2 --run mmppp_zdim2 --device 0
  ```
* To train IMMP++:

  ```
  python train.py --base_config configs/Robot/robot_manifold/base_config.yml --config configs/Robot/robot_manifold/immppp.yml --model.z_dim 2 --model.iso_reg 1 --run immppp_zdim2_reg1 --device 0
  ```
* Trained models are saved to [results/robot-manifold](results/robot-manifold).
* To see the trained results, open the [IPython Notebook](2-1-7D_robot_results_analysis.ipynb).
* To see the modulation of the latent values and viapoints, run the following command:

  ```
  python 2-2-7D_robot_modulation.py
  ```
* To see the online adpation, run the following command:

  ```
  python 2-3-7D_robot_obs_avoidance.py
  ```

### __3. SE(3) Pouring__

* To train MMP++:

  ```
  python train.py --base_config configs/SE3/mmppp/base_config.yml --config configs/SE3/mmppp/mmppp.yml --model.z_dim 2 --run mmppp_zdim2 --device 0
  ```

* Trained models are saved to [results/SE3/se3mmppp](results/SE3/se3mmppp).

* To see the modulation of the latent values and viapoints, run the following command:

  ```
  python 3-1-SE3_pouring_modulation.py
  ```

## Citation
If you found this library useful in your research, please consider citing:
```
@article{lee2024mmp++,
  title={MMP++: Motion Manifold Primitives With Parametric Curve Models},
  author={Lee, Yonghyeon},
  journal={IEEE Transactions on Robotics},
  year={2024},
  publisher={IEEE}
}
```

