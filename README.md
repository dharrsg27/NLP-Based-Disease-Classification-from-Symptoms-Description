# Comparing NLP Methods for Disease Classification from Symptoms Description: TF-IDF + KNN vs Fine-Tuned BERT Model

This project aims to perform disease classification from symptom descriptions using two different approaches:

1. **TF-IDF + KNN Classification**:
   - TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus). It transforms text data into numerical features by weighing terms based on their frequency and importance.
   - KNN (K-Nearest Neighbors) is a simple, instance-based learning algorithm. It classifies new data points based on the majority label of its k nearest neighbors in the feature space. When used with TF-IDF features, KNN classifies text documents based on their similarity to training examples.

2. **Fine-Tuned BERT Model**:
   - BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art transformer-based model developed by Google. It uses a deep learning architecture to understand the context of words in a sentence by looking at both their left and right contexts. BERT is pre-trained on large text corpora and can be fine-tuned for specific tasks like text classification, making it highly effective at capturing contextual relationships and nuances in text data.

## Dataset Description

The dataset consists of 1200 data compiled into a csv file with two columns: "label" and "text".

- label : contains the disease labels
- text : contains the natural language symptom descriptions.

The following 24 diseases are included in the dataset:

Psoriasis, Varicose Veins, Typhoid, Chicken pox, Impetigo, Dengue, Fungal infection, Common Cold, Pneumonia, Dimorphic Hemorrhoids, Arthritis, Acne, Bronchial Asthma, Hypertension, Migraine, Cervical spondylosis, Jaundice, Malaria, urinary tract infection, allergy, gastroesophageal reflux disease, drug reaction, peptic ulcer disease, diabetes

The dataset can be obtained from: https://www.kaggle.com/datasets/niyarrbarman/symptom2disease

## Workflow
1. Exploratory Data Analysis
   - Analyzed the frequency distribution of every disease
   - Created a word cloud for a disease to identify common words associated to the disease

2. Preprocessing (for the TF-IDF + KNN Classification)
   The text descriptions for each disease is preprocessed to prepare for the TF-IDF + KNN Classification in the following way:
   - Stopwords removal
   - Lemmatization
   - Punctuations removal and lowercasing of all texts

3. NLP Model Creation
   - TF-IDF + KNN Classification:
      - Performed TF-IDF vectorization
      - Performed cross-validation with Grid Search to determine the best possible parameters for the KNN model
   - BERT Model:
      - Loaded pretrained BERT model
      - Fine-tuned parameters for the BERT model for optimal training

4. Model Evaluation
   - **TF-IDF + KNN Classification:** 

     **Accuracy:** 93.611%
   - **Fine-Tuned BERT Model:** 

     **Accuracy:** 98.056%

The fine-tuned BERT model outperforms the TF-IDF + KNN approach, achieving an accuracy of 98.056% compared to 93.611% with the KNN classifier. This demonstrates the effectiveness of using state-of-the-art deep learning techniques, such as BERT, for complex text classification tasks. The BERT model's ability to understand context and nuances in symptom descriptions contributes to its higher accuracy.

On the other hand, TF-IDF + KNN Classification uses traditional text processing and machine learning methods. While effective, it may not capture contextual information as well as more advanced models.
