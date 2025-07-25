# Similarity & Siamese Neural Networks (SNNs)

This project explores how the choice of loss function influences the performance of Siamese Neural Networks (SNNs) in face verification tasks. As part of the seminar *Recent Advances in Machine Learning*, we compared three loss functions:

- **Contrastive Loss** (from the lecture)
- **Circle Loss**
- **Multi-Similarity Loss**

The models were trained using a ResNet-18-based Siamese architecture and evaluated on the **AT&T**, **CelebA**, and **VGGFace** datasets. Metrics such as AUROC, F1-Score, and Fisher Score were used to assess performance. Results showed that **Multi-Similarity Loss** consistently delivered the best results, especially on complex datasets.

**Team Members:**
- Haitham El Euch  
- Soroor Eskandari  
- Arman Niaruhi

---

# Setting Up the Conda Environment

To set up the required Conda environment, follow these steps:

1. **Download the dataset**  
   Download the dataset from [DATASET](https://drive.google.com/file/d/1oLIfYo21w744liIEN9EZi5i4Ej4shDQX/view?usp=share_link), and copy the entire `dataset` folder into the root directory of the project.


2. **Download the models**  
   Download our trained models [MODELS](https://drive.google.com/file/d/1ewfgJhApXS-5rkD-_ZjzqGMBO7dV6C6o/view?usp=sharing), and copy the entire `models` folder into the root directory of the project.

3. **Install Conda**  
   Make sure you have Conda installed. If not, you can download and install it from the [official Conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

4. **Set up the environment**  
   Open a terminal, navigate to the directory containing the `environment.yml` file, and run the following command:

   ```bash
   conda env create -f environment.yml
