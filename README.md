# Streamlit-based News Classification App-Team-EG5
<p align="center">
  <img src="https://github.com/Khululiwe/Streamlit_App-Team-EG5/blob/main/logo.png" alt="Logo" width="200"/>
</p>

## Overview

As Insights Lab Consulting, we have been engaged by Newstoday, an editorial company, to develop an automated news categorization system that can improve the efficiency and accuracy of content management. This project uses advanced machine learning and natural language processing techniques to automate the tagging and categorization of news articles. The end-to-end solution consists of gathering and preprocessing data, developing and assessing the model, and deploying it as a user-friendly web application with Streamlit. The system will enhance operational efficiency for the editorial team and improve content discovery for readers.

## Table of Contents
* [Project Overview](#project-overview)
* [Meet the Team](#meet-the-team)
* [Installation](#installation)
* [Usage Instructions](#usage-instructions)
* [Contact us](#contact-us)
  
## Meet the Team
Makhutjo Lehutjo (makhutjolehutjo8@gmail.com),
Khululiwe Hlongwane (khululiwe.hlongwane9966@gmail.com),
Nontuthuko Mpanza	(nontuthukompanza@outlook.com),
Asanda Gambu	(sinegugugambu@gmail.com),
Zandile Sibiya	(zandilepopo27@gmail.com),
Boitumelo Penyenye (happyboitumelo196@gmail.com)

## Installation
To get a local copy up and running, follow these simple steps:

### Prerequisites
- Python 3.7 or higher
- Anaconda for managing the environment

### Installation Steps
1. **Clone the repository:**
    ```bash
    git clone https://github.com/Khululiwe/Streamlit_App-Team-EG5.git
    cd Streamlit_App-Team-EG5
    ```

2. **Create a conda environment:**
    ```bash
    conda create --name classifierapp_env python=3.8
    ```
    
3. **Activate the conda environment:**
   ```bash
   conda activate classifierapp_env
   ```
   
5. **Install the required packages:**
    ```bash
    conda install -file requirements.txt
    ```

6. **Download necessary NLTK data:**
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## Usage Instructions
1. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2. **Input a news article:** Paste the text of the news article into the text area provided in the app.

3. **Classify the article:** Click the "Classify" button to get the predicted category of the news article.
