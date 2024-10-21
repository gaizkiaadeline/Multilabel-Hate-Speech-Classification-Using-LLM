# Multilabel Hate Speech Classification on Twitter Using Large Language Models
This project implements a multilabel classification model for hate speech detection using BERT and ALBERT pretrained models. The task is to classify social media posts into multiple labels that indicate hate speech (HS), abusive content, and specific types of hate speech (e.g., individual, group, race, relation, physical, gender, etc.).


The project uses a multilabel BERT-based classifier along with Data Augmentation techniques to improve the model's generalization. Additionally, it explores hyperparameter tuning (learning rate, batch size) to achieve optimal performance. The creative technique used here includes Easy Data Augmenter from the TextAttack library to enhance the training data, allowing the model to better handle variations in the text.

# Project Objectives:
  **Pretrained Models:**
  - The primary model used is ```bert-base-uncased```, implemented using ```BertForSequenceClassification``` for sequence classification with 9 output labels.
  - Additionally, the project incorporates two variations of the ALBERT model (```albert-base-v1``` and ```albert-base-v2```) for comparison and ensembling techniques.

  **Data Augmentation:**
  - The ```EasyDataAugmenter``` from ```TextAttack``` is applied to create augmented versions of the text data, improving the robustness of the classifier by exposing it to a wider range of linguistic variations.
  
  **Training and Evaluation:** 
  - Models are trained using the Trainer API from Hugging Face's transformers library, with evaluation metrics including accuracy, precision, recall, and F1-score.
  - Analyze these metrics to understand the model's strengths and weaknesses, and guide further improvements.

**Hyperparameter Tuning:**
- Learning Rate: The learning rate is tuned to optimize model convergence and generalization.
- Batch Size: Varying batch sizes are experimented with to find the best trade-off between performance and computational efficiency.

**Technologies Used:**
- Python: For coding and model implementation.
- Transformers (Hugging Face): To use pretrained language models.
- BERT (bert-base-uncased) and ALBERT (albert-base-v1 and albert-base-v2): The pretrained model used for the project.
- Scikit-learn: For calculating evaluation metrics.
- TensorFlow / PyTorch: For model training and tuning.

This project implements a multilabel classification model for detecting hate speech on social media using the pretrained BERT and ALBERT model. The project includes hyperparameter tuning, creative model architecture adjustments, and evaluation with metrics like accuracy, precision, recall, and F1-score.

![1111](https://github.com/user-attachments/assets/ab11e7de-d8ad-4da6-a83f-d817290536fd)

