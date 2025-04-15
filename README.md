# Deep Recommendation Systems on Amazon Reviews

This repository contains research implementations of deep recommendation systems utilizing Amazon reviews. It features multiple deep learning models that leverage review texts for rating prediction.

---

## 📌 Project Overview

This project explores various deep learning approaches to recommendation systems by integrating textual features with collaborative filtering techniques. The implemented models include:

### 🔹 **DeepCoNN**  
A dual Convolutional Neural Network (CNN) model that processes separate user and item reviews, combining their representations for rating prediction.

### 🔹 **HSACN**  
A hierarchical review embedding model that extracts n-gram features from reviews via convolutional operations while analyzing review statistics.

### 🔹 **NRCMA**  
A model incorporating cross-attention mechanisms and factorization machines to capture complex interactions between textual review features and identity embeddings.

---

## 📂 Dataset Details

The models are trained and evaluated using a subset of the publicly available **[Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)** from the McAuley-Lab. Specifically, the **"raw_review_Appliances"** subset is used for this project.

---

## 📁 Repository Structure

```
├── deep-rec-sys-amazon-reviews/
│   ├── pyproject.toml         # Project configuration and dependencies
│   ├── uv.lock                # Environment lock file
│   ├── assets/
│   │   └── images/            # Visual assets, plots, and figures
│   ├── eda/                   # Exploratory Data Analysis (EDA) notebooks
│   │   ├── Amazon_user_reviews_capstone.ipynb
│   │   ├── item_reviews.ipynb
│   │   └── recsys_preprocessing.ipynb
│   ├── modeling/              # Model implementations and training scripts
│   │   ├── DeepCoNN.ipynb
│   │   ├── DeepCoNN.py
│   │   ├── HSACN.ipynb
│   │   ├── HSACN.py
│   │   ├── NRCMA.ipynb
│   │   └── NRCMA.py
```
Each component is structured to separate data analysis from model development, ensuring clarity and ease of extension.

---

## 🚀 Installation and Setup

### 🔹 Clone the Repository
```bash
git clone https://github.com/your-username/deep-rec-sys-amazon-reviews.git
cd deep-rec-sys-amazon-reviews
```

### 🔹 Create a Virtual Environment
It is recommended to use Python 3.12 or later in a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 🔹 Install Dependencies
Install the required packages from `pyproject.toml` or `requirements.txt`:
```bash
pip install -r requirements.txt
```
Alternatively, manually install key dependencies:
```bash
pip install gensim>=4.3.3 matplotlib>=3.10.0 nltk>=3.9.1 numpy<2.0.0 \
            pandas>=2.2.3 scikit-learn>=1.6.1 surprise>=0.1 torch>=2.6.0 \
            torchinfo>=1.8.0 torchvision>=0.21.0 tqdm>=4.67.1 wandb>=0.19.6
```
Ensure your Python version meets the required (>=3.12) specification.

---

## ⚡ Running the Code

### 🔹 Exploratory Data Analysis (EDA)
Navigate to the `eda/` directory and open notebooks using Jupyter:
```bash
jupyter notebook eda/Amazon_user_reviews_capstone.ipynb
```
These notebooks cover preprocessing, data visualization, and statistical analysis of Amazon user reviews.

### 🔹 Model Training
Before training the model, the index tensor files need to be generated for users and items. To do that, run
```bash
python -m utils.final_data_loading_HSACN
```
Change the model name to get the files for each model respectively.

To train a model, navigate to the `deep-recsys-amazon-reviews` directory and run the corresponding Python script:
```bash
python -m modeling.DeepCoNN_train.py  # DeepCoNN
```
```bash
python -m modeling.HSACN_train.py     # HSACN
```
```bash
python -m modeling.NRCMA_train.py     # NRCMA
```

Each script handles data loading, preprocessing, training, evaluation, and model saving. Refer to notebooks for additional details on hyperparameters and configurations.

---

## 📚 Additional Resources
For a deeper understanding of the algorithms, evaluation metrics, and experimental setup, visit the project’s website:  
🔗 **[Project Website](https://recsys-user-reviews.github.io)**

---

## 🤝 Contributing
Contributions are welcome! If you’d like to improve the code, models, or documentation, feel free to open an issue or submit a pull request.

---

## 📜 License
This project is open source. Refer to the `LICENSE` file for details on usage and distribution rights.

---

🔹 **Happy Coding! 🚀**

