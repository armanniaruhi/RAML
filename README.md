# Similarity & Siamese Neural Networks (SNNs)

This project investigates how different loss functions impact the performance of **Siamese Neural Networks (SNNs)** for face verification tasks. As part of the course *Recent Advances in Machine Learning* at the university of siegen, we compared three loss functions that guide how similarity between face embeddings is learned:

- **Contrastive Loss** (classic distance-based loss using Euclidean distance)  
- **Circle Loss** (adaptive, cosine-based metric with margin control)  
- **Multi-Similarity (MS) Loss** (cosine-based, batch-level mining of hard pairs)

## Key Highlights

- **Architecture**: ResNet-18-based Siamese model  
- **Datasets**:  
  - AT&T (40 subjects, 400 grayscale images)  
  - CelebA (200K+ celebrity images)  
  - VGGFace (2.6K+ identities, 2M+ images)  
- **Evaluation Metrics**: AUROC, F1 Score, Precision, Recall, Fisher Score, Bhattacharyya Coefficient  
- **Best Performing Loss**: Multi-Similarity Loss – best generalization and class separability

---

## Summary of Findings

| Loss Function       | Accuracy (CelebA) | F1 Score (CelebA) | Fisher Score (AT&T) | Comments                      |
|---------------------|-------------------|-------------------|---------------------|-------------------------------|
| Contrastive         | 87.87%            | 86.98%            | 1.67                | Weakest on complex data       |
| Circle              | 100%              | 100%              | 6.87                | Strong performance, stable    |
| Multi-Similarity    | 99.43%            | 99.44%            | 9.52                | Best overall across datasets  |

### Visual Summary

- **t-SNE**: Circle and MS Loss yield compact, well-separated clusters  
- **ROC Curves**: MS Loss achieves the best AUROC across datasets  
- **Similarity Matrices**: Clearer separation for Circle and MS Loss  
- **Fisher Score**: Highest for MS Loss, indicating best class separation

---

## Methodology

- **Architecture**: Two-branch Siamese network with a shared ResNet-18 backbone  
- **Embedding Size**: 256-dimensional  
- **Image Preprocessing**:  
  - Resize to 100×100  
  - Convert to grayscale  
  - Apply random flipping, rotation, cropping, and noise for augmentation

- **Training Setup**:  
  - Optimizer: Adam, learning rate = 1e-4  
  - Batch size = 32, Epochs = 60  
  - Early stopping with patience = 3

- **Loss Settings**:  
  - Contrastive Loss: margin = 1.0  
  - Circle Loss: γ = 256, margin = 0.25  
  - MS Loss: α = 30, β = 80, margin = 0.5

- **Evaluation**:  
  - 5-fold cross-validation  
  - Metrics averaged across all folds  
  - Visualizations via t-SNE, ROC curves, similarity matrices

---

## Setting Up the Environment

### 1. Download the Dataset

Download the dataset from the following link:  
[Dataset (Google Drive)](https://drive.google.com/file/d/1l6xXrNVAUduUy4zl0eI0PSDIsQ7NZEpV/view?usp=sharing)

Place the extracted `dataset` folder in the root directory of the project.

### 2. Download the Pretrained Models

Download pretrained models from:  
[Models (Google Drive)](https://drive.google.com/file/d/1iXWGV09esjQExlD_GccUHg6Og7wCnjo0/view?usp=sharing)

Place the `models` folder in the root directory of the project.

### 3. Download our Results and Plots

Download some of the results of training and test:  
[Results (Google Drive)](https://drive.google.com/drive/folders/1t-HY0COdvQUTeS2G0BSizCZ-o0rfJAWe?usp=sharing)


### 4. Create the Virtual Environment and Activate it

create the venv

```bash
python -m venv raml2025 
```

In the terminal, activate our virtual environment to avoid extra package installation:

In linux/mac os:
```bash
source raml2025/bin/activate
pip install -r requirements.txt

```

In Windows:
```bash
raml2025\bin\activate
pip install -r requirements.txt

```


### 5.Configuration (`config.yml`)

The training and model settings are controlled via the `config.yml` file. You can modify this file to change the behavior of the Siamese Neural Network training process.

