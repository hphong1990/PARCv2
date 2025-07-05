<h1><a href="https://arxiv.org/abs/2402.12503">PARCv2: Physics-aware Recurrent Convolutional Neural Networks for Spatiotemporal Dynamics Modeling</a></h1>
<h3>Phong C.H. Nguyen, Xinlun Cheng, Shahab Azarfar, Pradeep Seshadri, Yen T. Nguyen, Munho Kim, Sanghun Choi, H.S. Udaykumar, Stephen Baek</h3>

**Published at:** *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*  
**Pages:** 37649-37666  
**Links:** [ICML Proceedings](https://proceedings.mlr.press/v235/nguyen24c.html) | [PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/nguyen24c/nguyen24c.pdf) | [OpenReview](https://openreview.net/forum?id=T0zR4mdSce) | [arXiv](https://arxiv.org/abs/2402.12503)

<h2> Key Contributions </h2>

- **Enhanced Architecture**: Extends the original PARC framework with differential operators specifically designed for advection-reaction-diffusion equations
- **Hybrid Integral Solver**: Incorporates a novel hybrid integral solver for stable, long-time predictions in challenging advection-dominated regimes
- **Differentiator-Integrator Design**: Inherits and improves upon the differentiator-integrator architecture from <a href="https://www.science.org/doi/10.1126/sciadv.add6868">physics-aware recurrent convolutions (PARC)</a>
- **Versatile Applications**: Demonstrates effectiveness across diverse physics problems from fluid dynamics to energetic materials
- **Benchmark Performance**: Comprehensive evaluation against physics-informed and learning bias models on standard benchmarks

<h2> Methodology Highlights </h2>

- **Inductive Bias Approach**: Leverages physics-aware inductive biases for better generalization to unseen dynamics
- **Spatiotemporal Modeling**: Specialized for learning spatiotemporal dynamics governed by advection-diffusion-reaction equations
- **Advection-Dominance**: Specifically targets unsteady, fast transient, and advection-dominated physics problems
- **Sharp Gradient Handling**: Capable of modeling evolving state fields with sharp gradients and rapidly deforming material interfaces
- **Long-term Stability**: Designed for stable predictions over extended time horizons

<h2> Applications and Results </h2>

PARCv2 has been extensively validated on multiple benchmark problems and real-world applications:

<h3> Standard Benchmark Problems </h3>

- **2D Burgers' Equation**: Nonlinear advection-diffusion dynamics with shock formation
- **2D Navier-Stokes Equations**: Unsteady fluid flow with complex vortical structures
- **Comparative Performance**: Demonstrated superior performance compared to physics-informed neural networks (PINNs) and other learning bias models

<h3> Complex Physics Applications </h3>

- **Shock-Induced Reaction Problems**: Modeling energetic materials under extreme conditions
- **Advection-Dominated Regimes**: Successful handling of challenging advection-dominant flows
- **Multi-Scale Dynamics**: Capable of capturing both fast transient and long-term evolution

<h2> Some examples from various benchmark problems </h2>
<h3> 2D Burgers' </h3>
<img src ="https://github.com/hphong1990/PARCv2/assets/22065833/289bb68a-ffd6-4c2a-8e12-139df17a6ead">
<h3> 2D Navier-Stokes for Unsteady Flow </h3>
<img src = "https://github.com/hphong1990/PARCv2/assets/22065833/c112d3e5-2865-448c-a9b2-6ff2298dd5de">

<h3> Energy localization of energetic materials </h3>
<img src = "https://github.com/hphong1990/PARCv2/assets/22065833/65fdb43d-c65b-44d1-8b33-a55f33790db2">

<h2> Requirements </h2>

**System Requirements:**
- Python 3.6.13 or higher
- TensorFlow 2.13.0 (for deep learning computations)
- CUDA-compatible GPU (recommended for training)

**Python Dependencies:**
- numpy (numerical computations)
- scipy (scientific computing)
- Pillow (image processing)
- pandas (data manipulation)
- matplotlib (visualization)
- scikit-image (image processing)
- scikit-learn (machine learning utilities)
- opencv-python (computer vision)

**Installation:**
```bash
pip install tensorflow==2.13.0 numpy scipy pillow pandas matplotlib scikit-image scikit-learn opencv-python
```

<h2> Dataset and Reproducibility </h2>

The datasets used in this work are available for download to ensure reproducibility of the results presented in the paper:

- **<a href="https://virginia.box.com/s/khrehgg574wm9r4b7qelu2jt1374kvtf">2D Burgers' Equation Dataset</a>**: Complete dataset for reproducing Burgers' equation experiments
- **Additional Datasets**: Contact the authors for access to Navier-Stokes and energetic materials datasets

**Usage:**
1. Download the dataset from the provided link
2. Extract the files to the appropriate directory
3. Run the demo notebooks in the `Demo/` folder to reproduce the results
4. Modify the data paths in the configuration files as needed

<h2> Getting Started </h2>

**Quick Start with Demo Notebooks:**
1. `Demo/parc_v2_burger.ipynb` - Burgers' equation demonstrations
2. `Demo/parc_v2_ns.ipynb` - Navier-Stokes equation examples  
3. `Demo/parc_v2_em.ipynb` - Energetic materials applications
4. `Demo/parc_v2_demo_hypersonic.py` - Hypersonic flow example

**Model Components:**
- `PARC/model/` - Core model implementations for different physics problems
- `PARC/data/` - Data loading and preprocessing utilities
- `PARC/visualization/` - Visualization tools for results analysis

<h2> Related Work and Baselines </h2>

This repository also includes baseline implementations for comparison studies:

**Fourier Neural Operator (FNO) Baselines:**
- `Related_Work (Baseline)/fno/` - FNO implementations for different physics problems
- Physics-Informed FNO (PIFNO) variants included
- Configuration files for reproducing baseline results

**PARC Neural ODE Comparisons:**
- `Related_Work (Baseline)/parc_neuralode/` - Neural ODE implementations
- Pre-trained models for direct performance comparison
- Jupyter notebooks for result reproduction


<h2> Citation </h2>
If you find PARCv2 helpful, please consider citing us with：

```
@InProceedings{pmlr-v235-nguyen24c,
  title = 	 {{PARC}v2: Physics-aware Recurrent Convolutional Neural Networks for Spatiotemporal Dynamics Modeling},
  author =       {Nguyen, Phong C.H. and Cheng, Xinlun and Azarfar, Shahab and Seshadri, Pradeep and Nguyen, Yen T. and Kim, Munho and Choi, Sanghun and Udaykumar, H.S. and Baek, Stephen},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {37649--37666},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/nguyen24c/nguyen24c.pdf},
  url = 	 {https://proceedings.mlr.press/v235/nguyen24c.html},
}

```
