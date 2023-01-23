<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/adithyagaurav/Segformer3D">
    <img src="results/UMD_logo.jpeg" alt="Logo" width="160" height="80">
  </a>

<h3 align="center">SEGFORMER3D</h3>

  <p align="center">
    Multi task vision transformer for segmentation and depth estimation
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Segformer3D Demo][product-screenshot]]()

Welcome to the enhanced Segformer 3D : A Vision Transformer neural network for segmentation and depth estimation using attention mechanism.

This project implements the original Segformer architecture specialized in image segmentation and enhances it to be a multi-task network which also predicts depth, making it Segformer3D. The aim of the project is to optimize an existing neural network which uses attention mechanism to execute multiple tasks. Multi-task neural networks hold significant advantages in systems with low latency requirements.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python-badge]][Python-url]
* [![PyTorch][PyTorch-badge]][PyTorch-url]
* [![Numpy][Numpy-badge]][Numpy-url]
* [![Matplotlib][Matplotlib-badge]][Matplotlib-url]
* [![OpenCV][OpenCV-badge]][OpenCV-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Architecture

The segformer network uses an encoder-decoder style architecture, with attention modules integrated in the encoder block. Attention mechanism, in its simplest form works by looking at different parts of the image and deciding which parts are most important for understanding the overall picture. It does this by comparing different parts of the image to each other using something called a "key" and a "query". The key is like a label that describes what is in each part of the image, and the query is like a question that the model is trying to answer. By comparing the key and the query, the attention mechanism can figure out which parts of the image are most important for answering the question.

<div align="center">
  <a href="https://github.com/adithyagaurav/Segformer3D">
    <img src="results/segformer_architecture.png" alt="Logo">
  </a>
</div>


In the original paper Segformer [[1]](https://arxiv.org/pdf/2105.15203.pdf), the authors describe network to be a cascaded sequence of transformer and downsampling block. The transformer block is an ensemble responsible for implementing attention mechanism.

1. Overlap patch embedding: This module is responsible for dividing the input image into smaller blocks for attention mechanism to focus on independently and estimate important features.
2. Efficient self attention : This module uses query, key and value vectors to extract features from different parts of image in parallel and assigns them weights on the basis of how relevant they are to understanding the context of the image.
3. Overlap patch merging : This module combines the output of multiple transformer blocks into a single output. It does so by overlaying the outputs of the different transformer blocks on top of each other, with a certain amount of overlap between the blocks.
4. Mix FFN : This module is used to process the output of the transformer blocks and make final predictions based on this output. 

The decoder module is a straightforward MLP architecture that upsamples the encoded vector by utilizing features from hidden layers of the encoder.

This project modifies the original Segformer architecture to integrate an additional decoder block that shares the encoded vector to upsample it to produce a depth estimation map. Furthermore, the loss function is required to be modified from Cross Entropy loss to Cross Entropy + MSELoss.

### Results

Carrying out Inference on a single image 

<div align="center">
  <a href="https://github.com/adithyagaurav/Segformer3D">
    <img src="results/out.png" alt="Logo" width="100" height="80">
  </a>
</div>

This architecture is a Vision Transformer at its core using attention mechanism. Visualizing the output from difference attention blocks at different stages helps in a gaining a better understanding of how the model is recognizing the important features which need to focussed on. Following is the vizualization from attention layers.

<div align="center">
  <a href="https://github.com/adithyagaurav/Segformer3D">
    <img src="results/viz.gif" alt="Logo">
  </a>
</div>

<!-- GETTING STARTED -->
## Getting Started

The dataset used in this project is the Cityscapes dataset.

Download the dataset from [link](https://drive.google.com/drive/folders/16wql9YhBGNuXt2c_xk8cWX8z-gqNgr_s?usp=share_link) and place it in the `data/` folder

Download the weights from [link](https://drive.google.com/file/d/1MY9JbKJ3mmx-fE1sc2rQG-76tClH9_q6/view?usp=share_link) and place it in the `weights/` folder

Download the test video from [link](https://drive.google.com/file/d/1vTAh8DTrzBtDs69vuqa5l4cChHwDxgzL/view?usp=share_link) and place it in the `video/` folder

Download the backbone pertained weights from [link](https://drive.google.com/file/d/1rDu2DQO42PV3pAYjb6CuhANrrxt8X0yX/view?usp=share_link)

### Prerequisites

* Python & pip
  ```sh
  sudo apt-get install python3.6
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python get-pip.py
  ```
* PyTorch
  ```sh
  pip install torch
  pip install torchvision
  ```
* Numpy, Matplotlib, Einops, Timm
  ```sh
  pip install numpy
  pip install matplotlib
  pip install einops
  pip install timm
  pip install opencv-python
  ```


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/adithyagaurav/Segformer3D.git
   ```
2. To run inference on a single image
  ```sh
  python3 src/test.py --weights <path to weights downloaded> --image_dir <path to folder containing the image>
  ```
3. To run inference on video
  ```sh
  python3 src/test_video.py --weights <path to weights downloaded> --video <path to the inference video>
  ```
4. To train the network using pretrained imagenet weights
  ```sh
  python3 src/train.py --weights <path to weights downloaded> --data_dir <path to data_seg_depth folder downloaded>
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Adithya Singh - agsingh@umd.edu

Project Link: [https://github.com/adithyagaurav/Segformer3D](https://github.com/adithyagaurav/Segformer3D)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, and Ping Luo. 2021. SegFormer: Simple
and efficient design for semantic segmentation with transformers](https://arxiv.org/abs/2105.15203)
* [Segformer Exloration by ThinkAutonomous](https://courses.thinkautonomous.ai/segformers)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/adithyagaurav/Segformer3D.svg?style=for-the-badge
[contributors-url]: https://github.com/adithyagaurav/Segformer3D/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/adithyagaurav/Segformer3D.svg?style=for-the-badge
[forks-url]: https://github.com/adithyagaurav/Segformer3D/network/members
[stars-shield]: https://img.shields.io/github/stars/adithyagaurav/Segformer3D.svg?style=for-the-badge
[stars-url]: https://github.com/adithyagaurav/Segformer3D/stargazers
[issues-shield]: https://img.shields.io/github/issues/adithyagaurav/Segformer3D.svg?style=for-the-badge
[issues-url]: https://github.com/adithyagaurav/Segformer3D/issues
[license-shield]: https://img.shields.io/github/license/adithyagaurav/Segformer3D.svg?style=for-the-badge
[license-url]: https://github.com/adithyagaurav/Segformer3D/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: results/out.gif
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
[Python-badge]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[NumPy-badge]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[Numpy-url]: https://numpy.org/
[PyTorch-badge]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[scikit-learn-badge]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/stable/
[Pandas-badge]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/
[Matplotlib-badge]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Matplotlib-url]: https://matplotlib.org/
[OpenCV-badge]: https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white
[OpenCV-url]: https://opencv.org/