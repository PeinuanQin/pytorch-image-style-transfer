# Citation

If you find this code useful for your research, please cite the following:
```shell
@misc{
  author = {PeinuanQin, YitianYang},
  title = {pytorch-img-style-transfer},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gordicaleksa/pytorch-neural-style-transfer}},
}
```

- our code is reproduced based on the work of https://github.com/gordicaleksa/pytorch-neural-style-transfer
- but we make it more clear, annotated and more object-oriented


## Neural Style Transfer (optimization method) :computer: + :art: = :heart:
This repo contains a concise PyTorch implementation of the original NST paper (:link: [Gatys et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)).

### What is NST algorithm?
The algorithm transfers style from one input image (the style image) onto another input image (the content image) using CNN nets (usually VGG-16/19) and gives a composite, stylized image out which keeps the content from the content image but takes the style from the style image.

<p align="center">
<img src="data/content_images/golden_gate.jpg" width="570"/>
<img src="data/output_images/golden_gate.jpg_vg_houses.jpg_0999..jpg" width="570"/>
</p>

### Why use our repo?
- We try best to use Object-oriented programming for coding reproduction, codes are clean and readable
- Annotate in great detail, almost for **every line**!!
## Examples

Transfering style gives beautiful artistic results:

<p align="center">
<img src="data/style_images/vg_houses.jpg" height="202px">
<img src="data/output_images/golden_gate.jpg_vg_houses.jpg_0999..jpg" width="270px">
<img src="data/style_images/vg_starry_night.jpg" width="320px">
<img src="data/output_images/golden_gate.jpg_vg_starry_night.jpg_0900..jpg" width="270px">
</p>

### Content/Style tradeoff

Changing style weight gives you less or more style on the final image, assuming you keep the content weight constant. <br/>
I did increments of 10 here for style weight (1e1, 1e2, 1e3, 1e4), while keeping content weight at constant 1e5, and I used random image as initialization image. 

<p align="center">
<img src="examples/style-tradeoff/figures_vg_starry_night_o_lbfgs_i_random_h_352_m_vgg19_cw_100000.0_sw_10.0_tv_1.0_resized.jpg" width="200px">
<img src="examples/style-tradeoff/figures_vg_starry_night_o_lbfgs_i_random_h_352_m_vgg19_cw_100000.0_sw_100.0_tv_1.0_resized.jpg" width="200px">
<img src="examples/style-tradeoff/figures_vg_starry_night_o_lbfgs_i_random_h_352_m_vgg19_cw_100000.0_sw_1000.0_tv_1.0_resized.jpg" width="200px">
<img src="examples/style-tradeoff/figures_vg_starry_night_o_lbfgs_i_random_h_352_m_vgg19_cw_100000.0_sw_10000.0_tv_1.0_resized.jpg" width="200px">
</p>

### Impact of total variation (tv) loss

Rarely explained, the total variation loss i.e. it's corresponding weight controls the smoothness of the image. <br/>
I also did increments of 10 here (1e1, 1e4, 1e5, 1e6) and I used content image as initialization image.

<p align="center">
<img src="examples/tv-tradeoff/figures_candy_o_lbfgs_i_content_h_350_m_vgg19_cw_100000.0_sw_30000.0_tv_10.0_resized.jpg" width="200px">
<img src="examples/tv-tradeoff/figures_candy_o_lbfgs_i_content_h_350_m_vgg19_cw_100000.0_sw_30000.0_tv_10000.0_resized.jpg" width="200px">
<img src="examples/tv-tradeoff/figures_candy_o_lbfgs_i_content_h_350_m_vgg19_cw_100000.0_sw_30000.0_tv_100000.0_resized.jpg" width="200px">
<img src="examples/tv-tradeoff/figures_candy_o_lbfgs_i_content_h_350_m_vgg19_cw_100000.0_sw_30000.0_tv_1000000.0_resized.jpg" width="200px">
</p>

### Optimization initialization

Starting with different initialization images: noise (white or gaussian), content and style leads to different results. <br/>
Empirically content image gives the best results as explored in [this research paper](https://arxiv.org/pdf/1602.07188.pdf) also. <br/>
Here you can see results for content, random and style initialization in that order (left to right):

<p align="center">
<img src="examples/init_methods/golden_gate_vg_la_cafe_o_lbfgs_i_content_h_500_m_vgg19_cw_100000.0_sw_30000.0_tv_1.0_resized.jpg" width="270px">
<img src="examples/init_methods/golden_gate_vg_la_cafe_o_lbfgs_i_random_h_500_m_vgg19_cw_100000.0_sw_1000.0_tv_1.0_resized.jpg" width="270px">
<img src="examples/init_methods/golden_gate_vg_la_cafe_o_lbfgs_i_style_h_500_m_vgg19_cw_100000.0_sw_10.0_tv_0.1_resized.jpg" width="270px">
</p>

You can also see that with style initialization we had some content from the artwork leaking directly into our output.


### Content reconstruction

If we only use the content (perceptual) loss and try to minimize that objective function this is what we get (starting from noise):

<p align="center">
<img src="examples/content_reconstruction/0000.jpg" width="200px">
<img src="examples/content_reconstruction/0026.jpg" width="200px">
<img src="examples/content_reconstruction/0070.jpg" width="200px">
<img src="examples/content_reconstruction/0509.jpg" width="200px">
</p>

### Style reconstruction

We can do the same thing for style (on the left is the original art image "Candy") starting from noise:

<p align="center">
<img src="examples/style_reconstruction/candy.jpg" width="200px">
<img src="examples/style_reconstruction/0045.jpg" width="200px">
<img src="examples/style_reconstruction/0129.jpg" width="200px">
<img src="examples/style_reconstruction/0510.jpg" width="200px">
</p>

## Setup

```shell
conda create -n style python=3.7
conda activate style
cd style_transfer
pip install -r requirements.txt
python train.py
```

for pytorch installation, you can search in https://pytorch.org/get-started/previous-versions/ for more details
![img.png](examples/img.png)
## Usage

1. Copy content images to the default content image directory: `/data/content_images/`
2. Copy style images to the default style image directory: `/data/style_images/`
3. Set all the parameters in local_config files
4. Run `python train.py`

For more advanced usage take a look at the code it's (hopefully) self-explanatory

### Debugging/Experimenting

Q: L-BFGS can't run on my computer it takes too much GPU VRAM?<br/>
A: Set Adam as your default and take a look at the code for initial style/content/tv weights you should use as a start point.

Q: Output image looks too much like style image?<br/>
A: Decrease style weight or take a look at the table of weights (in neural_style_transfer.py), which I've included, that works.

Q: There is too much noise (image is not smooth)?<br/>
A: Increase total variation (tv) weight (usually by multiples of 10, again the table is your friend here or just experiment yourself).


## Acknowledgements
- thanks for the https://github.com/gordicaleksa/pytorch-neural-style-transfer
