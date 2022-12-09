# Alt-text generator for images
## Setup
### Handling big files
Since there are many big files in this directory (the model weights, for example), those are managed by git LFS.
- Before cloning the directory, use `brew install git-lfs` to set up git LFS for the user account.
- Then the directory should be good to clone!

### On MacBook Pro with M1 Pro chip:
- Install [Anaconda](https://docs.anaconda.com/anaconda/install/mac-os/)
- Create a python 3.8 environment using `conda create --name alt-text python==3.8`
- Activate the environment using `conda activate alt-text`
- Do `pip install -r requirements.txt` to install all the necessary packages
- Start Jupyter Lab server and run the notebook, should be good to go!

# Experiments
## model1
Weights are stored in `models` folder.
Architecture uses the same one as the notebook in Colab. But only have weights for the first 6 epochs.
## model2
Weights are stored in `models2` folder.
Architecture uses the same one as the notebook in Colab, with full 10 epochs.
## model3
Weights are stored in `models3` folder.
Uses bigger network includes multiple layers of LSTM and dropout.
## model4
Weights are stored in `models4` folder. Incorporates pretrained Glove embeddings.
## model5
Weights are stored in `models5` folder.
Incorporates pretrained ImageNet weights for the Xception feature extractor as well as Glove embeddings.