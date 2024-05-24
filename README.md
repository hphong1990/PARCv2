<h1><a href="https://arxiv.org/abs/2402.12503">PARCv2: Physics-aware Recurrent Convolutional Neural Networks for Spatiotemporal Dynamics Modeling</a></h1>
<h3>Phong C.H. Nguyen, Xinlun Cheng, Shahab Azarfar, Pradeep Seshadri, Yen T. Nguyen, Munho Kim, Sanghun Choi, H.S. Udaykumar, Stephen Baek</h3>

<h2> Highlights </h2>

- Differentiator-Integrator architecture inherited from the recent physics-aware recurrent convolutions <a href="https://www.science.org/doi/10.1126/sciadv.add6868">(PARC)</a>
- Learning the spatiotemporal dynamics governed by the advection-diffusion-reaction equation
- Aiming for unsteady, fast transient, and advection-dominated physics problems 
- Validated with various benchmark dynamic problems with varying level of advection-dominance.

<h2> Some examples from various benchmark problems </h2>
<h3> 2D Burgers' </h3>
<p align="center">
<video  src= "https://github.com/hphong1990/PARCv2/assets/22065833/085f021d-5d2b-47cd-a9e9-07591cc6b64b"></video>
</p>

<h3> 2D Navier-Stokes for Unsteady Flow </h3>
(to be cont.)
<h3> 2D Supersonic flow </h3>
(to be cont.)
<h3> Energy localization of energetic materials </h3>
(to be cont.)

<h2> Requirements </h2>

- python 3.6.13
- tensorflow 2.13.0
- numpy
- scipy
- Pillow
- pandas
- matplotlib 
- scikit-image
- scikit-learn
- opencv-python

<h2> Dataset </h2>
The required data to reproduce the result presented in the paper can be downloaded using the below link:

- <a href = "https://virginia.box.com/s/khrehgg574wm9r4b7qelu2jt1374kvtf"> 2D Burgers' Equation  </a>
- <a href = "https://virginia.box.com/s/4zot7jo32x0fzxb2pg3yv7t18y4lfdlm"> Navier Stokes Equation </a>
- <a href = "https://virginia.box.com/s/khrehgg574wm9r4b7qelu2jt1374kvtf"> Energy Localization in Energetic Materials </a>

<h2> Tutorials </h2>
(to be cont.)

<h2> Citation </h2>
If you find PARCv2 helpful, please consider citing us with：

```
@misc{nguyen2024parcv2,
      title={PARCv2: Physics-aware Recurrent Convolutional Neural Networks for Spatiotemporal Dynamics Modeling}, 
      author={Phong C. H. Nguyen and Xinlun Cheng and Shahab Azarfar and Pradeep Seshadri and Yen T. Nguyen and Munho Kim and Sanghun Choi and H. S. Udaykumar and Stephen Baek},
      year={2024},
      eprint={2402.12503},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
