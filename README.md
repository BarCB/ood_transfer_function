# ood_transfer_function


## Prerequisites

The following packages are necessary:

<ul>
  <li>albumentations 1.2.0</li>
  <li>torch 1.12.0</li>
  <li>CUDA Toolkit 11.6</li>
  <li>fastai 2.7.4</li>
</ul>


## Installation Steps

<ol>
  <li>If a GPU cuda ready is available:</li>
  <ol>
    <li>Install CUDA Toolkit 11.6 from https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local</li>
    <li>Run the command pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116</li>
  </ol>
  <li>If no GPU available:</li>
  <ol>
    <li>Run the command pip3 install torch torchvision torchaudio</li>
  </ol>
  <li>Install albumentation with the command pip install albumentations</li>
  <li>Install fastai with the command pip install fastai</li>
</ol>