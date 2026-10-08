# 👁️👄 DeficiVision: Intelligent Nutritional Health Assessment

An AI-powered diagnostic prototype utilizing a hybrid ResNet50 (CNN) and Vision Transformer (ViT) pipeline to detect vitamin deficiencies from clinical images of eyes, lips, tongues, gums, skin, and hair. 

## Features
* **Hybrid Architecture:** Combines the spatial feature extraction of CNNs with the global context understanding of Vision Transformers.
* **Soft-Voting Ensemble:** Averages probability maps to output high-confidence predictions.
* **Interactive UI:** Built with Streamlit for a clean, clinical-style dashboard including dynamic health recommendations.


**Download the Dataset**
The clinical image dataset used to train and test this model is hosted on Google Drive. 
* Download the dataset here: https://drive.google.com/drive/folders/1jdH91I3-OIAt8_OEoTI3Ros9SlsruBNv?usp=sharing
  
*Note: Extract the downloaded folder into the root directory of this project before running any training scripts.*

Data Preprocessing:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]https://colab.research.google.com/drive/1IH-GrV-vws17cKrSW70hU1Y0RCOphOBD?usp=sharing

CNN_Training:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]https://colab.research.google.com/drive/1cvr1cGbCkZBtGjP05Co5EodL0rnc9201?usp=sharing

ViT_Training:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]https://colab.research.google.com/drive/1padvTC3i6JZ9o9dKjPdTUh8HFhweMEuI?usp=sharing

Hybrid_Integration:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]https://colab.research.google.com/drive/1YBbPkAjHKXMHgfCKOWctcwndS5ft29wF?usp=sharing

## Installation & Setup

1. Clone the repository
```bash
git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
cd YOUR-REPO-NAME

2. Install dependencies
pip install -r requirements.txt

3. Download the Model Weights
Because the trained .pth files are too large for standard GitHub hosting, they are hosted securely on Google Drive.

Download cnn_vitamin_model.pth here: https://drive.google.com/file/d/1GXDhdwP2Xmwx_nUt5aYi8Mp4GJ4a_Dqt/view
Download vit_vitamin_model.pth here: https://drive.google.com/file/d/1vHg0QHWgEJJujWsEpD50mdKe5MxpxCHr/view

4. Run the Application
python -m streamlit run app.py


