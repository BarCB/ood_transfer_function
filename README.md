# ood_transfer_function


## Prerequisites

The following packages are necessary:

<ul>
  <li>albumentations 1.2.0</li>
  <li>torch 1.12.0</li>
  <li>torchvision 0.13.0</li>
  <li>torchaudio 0.12.0</li>
  <li>CUDA Toolkit 11.6</li>
  <li>fastai 2.7.4</li>
  <li>numpy 1.23</li>
</ul>

## Installation Steps

<ol>
  <li>Install Anaconda</li>
  <li>Create an environment with Python 3.10</li>
  <li>If a GPU cuda ready is available:</li>
  <ol>
    <li>Install CUDA Toolkit 11.6 from https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local</li>
    <li>Run the command <code>conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge</code></li>
  </ol>
  <li>If no GPU available:</li>
  <ol>
    <li>Run the command <code>pip3 install torch torchvision torchaudio</code></li>
  </ol>
  <li>Install albumentation with the command <code>pip install albumentations</code></li>
  <li>Install fastai with the command <code>pip install fastai==2.7.8</code></li>
</ol>