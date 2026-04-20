# DSAI 490 - Assignment 1: Representation Learning with Autoencoders (AE & VAE)

**Author:** Abdalla Tamer  
**Student ID:** 202201240  

## 📌 Project Overview
This project explores unsupervised representation learning by implementing two fundamental deep learning architectures: a **Standard Autoencoder (AE)** and a **Variational Autoencoder (VAE)**. 

Using the **Medical MNIST** dataset, this codebase demonstrates how to compress high-dimensional medical images into lower-dimensional latent spaces and reconstruct them. Furthermore, it highlights the key differences between deterministic mapping (AE) and probabilistic generative modeling (VAE), showcasing applications like image denoising and the generation of novel synthetic medical images.

## 📂 Project Structure
Following the course code conventions, the repository is structured modularly:

├── notebooks/
│   └── experiment.ipynb # Main training, evaluation, and visualization pipeline
├── src/
│   ├── __init__.py
│   ├── data_processing.py # tf.data pipeline and Kaggle download logic
│   └── model.py           # TensorFlow subclassed AE and VAE architectures
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

## 🚀 How to Run the Project

### Prerequisites

  * Python 3.8+ (64-bit recommended)
  * A Kaggle account (to download the dataset via the `opendatasets` library)

### Setup Instructions

1.  **Clone the repository:**

    ```bash
      git clone https://github.com/AbdallaTamer/GENAI
    cd GENAI
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Experiment Notebook:**

      * Open Visual Studio Code or Jupyter Notebook.
      * Navigate to the `notebooks/` directory and open `experiment.ipynb`.
      * **Note on Data:** When you run the first cell that loads the data, `opendatasets` will prompt you for your Kaggle Username and Kaggle Key. You can find these by going to your Kaggle Account settings and clicking "Create New API Token" to download your `kaggle.json` file.
      * Run the notebook sequentially to train the models, view latent space reconstructions, test denoising, and generate new images.

## 🎥 Video Demonstration

A brief 2–5 minute video demonstration covering model training, latent space visualizations, generated samples, and key findings can be found below:

👉 **[Watch the Video Demonstration Here (Google Drive)](https://drive.google.com/file/d/1K8qSBOG0tIO9w0miXlGu-TyOdz_B9fPn/view?usp=sharing)**

## 🛠️ Built With

  * **TensorFlow & Keras:** For model architecture and deep learning pipelines.
  * **tf.data:** For highly optimized, asynchronous data loading and preprocessing.
  * **Matplotlib:** For latent space and reconstruction visualization.
