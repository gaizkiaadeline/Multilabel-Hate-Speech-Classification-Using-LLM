# Multilabel Hate Speech Classification on Social Media Using Large Language Models
Hate speech on social media is a significant concern, and automatically detecting it helps relevant stakeholders take efficient actions. In this project, we build a multilabel classification model to detect various forms of hate speech, leveraging Large Language Models (LLMs). The classification model identifies the following labels from social media text data:

Hate Speech (HS)
- Abusive
- HS_Individual
- HS_Group
- HS_Relation
- HS_Race
- HS_Physical
- HS_Gender
- HS_Other

**Project Objectives:**
  **Model Development:**
- Utilize pretrained BERT (bert-base-uncased) for multilabel hate speech classification.
- Perform hyperparameter tuning on at least two key parameters to optimize model performance.
- Leverage MultilabelBERT, a custom extension of the BERT model tailored for multilabel classification, supporting multiple hate speech categories.

  **Evaluation:**
- Present results including accuracy, precision, recall, and F1-score for training, validation, and testing datasets.
- Analyze these metrics to understand the model's strengths and areas for improvement.

  **Creative Techniques:**
  
Implement innovative techniques, such as customizing the BERT architecture for multilabel classification and adjusting the loss function to handle imbalanced data. These techniques aim to improve the model's ability to differentiate between multiple hate speech labels simultaneously.

**Key Features:**
- Large Language Model: The project uses the pretrained BERT (bert-base-uncased) model, fine-tuned for multilabel hate speech detection tasks.
- Multilabel Classification: Simultaneous classification of multiple hate speech labels to cover a broad spectrum of hate speech categories.
- Evaluation Metrics: The project is evaluated on standard classification metrics, including accuracy, precision, recall, and F1-score.
- Hyperparameter Tuning: Fine-tuning of at least two hyperparameters (e.g., learning rate, batch size) to improve the model’s generalization and performance.

**Technologies Used:**
- Python: For coding and model implementation.
- Transformers (Hugging Face): To use pretrained language models.
- BERT (bert-base-uncased): The pretrained model used for the project.
- Scikit-learn: For calculating evaluation metrics.
- TensorFlow / PyTorch: For model training and tuning.

This project implements a multilabel classification model for detecting hate speech on social media using the pretrained BERT (bert-base-uncased) model. The project includes hyperparameter tuning, creative model architecture adjustments, and evaluation with metrics like accuracy, precision, recall, and F1-score.
