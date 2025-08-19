# Fraudulent Claim Prediction Classification

This repository implements a machine learning solution to classify insurance claims as either fraudulent or legitimate. The project leverages various data preprocessing techniques, multiple machine learning models, and feature engineering methods to accurately identify potentially fraudulent claims in the automotive industry.

The solution is demonstrated using Jupyter notebooks alongside supporting data files and documentation, including presentation materials and datasets.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Notebooks Overview](#notebooks-overview)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Data Description](#data-description)
- [Results and Evaluation](#results-and-evaluation)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The Fraudulent Claim Prediction Classification project focuses on detecting fraudulent claims in the automotive industry. By analyzing historical claim data, the repository demonstrates how to leverage machine learning techniques—from data preprocessing to model evaluation—to distinguish between legitimate and fraudulent claims.

Key components of the project include:
- **Data Cleaning & Preprocessing:** Address missing values, normalize data, and encode categorical variables.
- **Feature Engineering:** Develop new features to improve model performance.
- **Model Training & Evaluation:** Compare multiple classifiers (e.g., Random Forest, Gradient Boosting) using performance metrics such as accuracy, precision, recall, and F1-score.
- **Documentation & Reporting:** Detailed analysis provided in notebooks, supported by presentation slides and PDF documentation.

## Project Structure

Below is the directory structure of the repository:

- **Cleaned/**  
  Contains cleaned and preprocessed datasets.

- **TestData/**  
  Test datasets used for evaluation.

- **TrainData/**  
  Training datasets used for model development.

- **final hack/**  
  Final implementation scripts and experiments.

- **Notebooks:**  
  - `Claim.ipynb`: Main notebook for claim prediction analysis.  
  - `Train_Demographics.ipynb`: Analysis focusing on training demographics data.  
  - `combined.ipynb`: Integration of various models and methods.  
  - `demo.ipynb`: Demonstration of the prediction pipeline.  
  - `final.ipynb`: Final version of the predictive model.  
  - `policy.ipynb`: Notebook focusing on policy-related features impacting claim outcomes.

- **Additional Files:**  
  - `Detecting Fraudulent Claims in Automotive Industry By Leveraging.pdf` and  
    `Detecting Fraudulent Claims in Automotive Industry By Leveraging.pptx`: Documentation and presentation materials describing methods and results.  
  - `ML types.png`: Visual representation of machine learning types or methodologies used in this project.  
  - `Test Data.zip` and `TrainData.zip`: Archived versions of the test and train datasets.  
  - CSV files (`lgb1.csv`, `rfc_os.csv`): Data outputs or logs from particular model experiments.

## Notebooks Overview

- **Claim.ipynb:**  
  Provides an end-to-end analysis of claim data, from loading and cleaning the data to model training and prediction.

- **Train_Demographics.ipynb:**  
  Focuses on exploring and analyzing demographic data from the claim records.

- **combined.ipynb:**  
  Integrates multiple machine learning approaches to compare their performance in fraud detection.

- **demo.ipynb:**  
  Offers a streamlined demonstration of the prediction process, suitable for presentations or quick evaluations.

- **final.ipynb:**  
  Contains the final iteration of the model along with comprehensive evaluation metrics.

- **policy.ipynb:**  
  Analyzes policy details and their relationship with fraudulent claims.

## Installation and Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/manuml1312/Fraudulent-Claim-Prediction_Classification.git
   cd Fraudulent-Claim-Prediction_Classification
   ```

2. **Set up a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- **Running Notebooks:**  
  Open the notebooks using Jupyter Notebook or JupyterLab:
  ```bash
  jupyter notebook
  ```
  Explore each notebook file to understand data preprocessing, feature engineering, and model training processes.

- **Data Preparation:**  
  Ensure that the data folders (`Cleaned`, `TestData`, `TrainData`) and zip files are in place, as they are used by the notebooks for training and evaluation.

## Data Description

The project uses claim data from the automotive industry, containing features such as claimant demographics, vehicle details, claim amounts, and policy specifics. The data is split into training and testing sets to facilitate robust model evaluation.

## Results and Evaluation

Model performance is evaluated using several metrics, including:
- Accuracy
- Precision
- Recall
- F1-Score

Evaluation results, confusion matrices, and ROC curves are detailed within the notebooks, particularly in `final.ipynb` and `combined.ipynb`.

## Future Enhancements

- **Model Optimization:** Experiment with additional algorithms and hyperparameter tuning.
- **Deployment Pipeline:** Develop a scalable deployment framework for real-time claim prediction.
- **Advanced Feature Engineering:** Incorporate more complex features and external data sources.
- **User Interface:** Implement a web interface or API for interactive predictions.

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks to all contributors and the open-source community for providing the tools and libraries essential to developing this project. Additional thanks to the research and industry partners who provided insights into automotive claim processing.
