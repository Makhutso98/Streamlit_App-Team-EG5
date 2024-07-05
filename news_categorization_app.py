"""
    Streamlit webserver-based News Categorization Engine.

    Author: Insights Lab Consulting.

    Description: This file is used to launch a Streamlit web application for 
    automated news categorization using machine learning models.

    For further help with the Streamlit framework, see:
    https://docs.streamlit.io/en/latest/

"""

# Streamlit dependencies
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

# Load necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load your vectorizer and model
vectorizer_path = "tfidf_vectorizer.pkl"
with open(vectorizer_path, "rb") as file:
    vectorizer = joblib.load(file)

model_path = "neural_network_model.pkl"
with open(model_path, "rb") as file:
    model = joblib.load(file)

# Customize the stop words list for news-specific stop words
stop_words = set(stopwords.words('english'))
news_specific_stop_words = {"said", "mr", "mrs", "one", "two", "new", "first", "last", "also"}
stop_words_list = list(stop_words.union(news_specific_stop_words))

lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A).lower()
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words_list]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# App declaration
def main():
    page_options = ["News Classifier", "Solution Overview", "About", "Contact Us", "For The Tech Geeks", "FAQ"]

    page_selection = st.sidebar.selectbox("Choose Option", page_options)

    if page_selection == "News Classifier":
        st.title("Newstoday Automated News Categorization")
        st.subheader("Transforming editorial workflows with Insights Lab Consulting")

        options = ["Prediction", "Information"]
        selection = st.sidebar.selectbox("Choose Option", options)

        if selection == "Information":
            st.info("General Information")
            st.markdown("""
                Welcome to the Newstoday Automated News Categorization system. 
                Our goal is to help Newstoday efficiently categorize news articles using advanced machine learning techniques. 
                This system will transform editorial workflows, making them faster and more accurate.
            """)

        if selection == "Prediction":
            st.info("Prediction with ML Models")
            news_text = st.text_area("Enter Text", "Type Here")

            if st.button("Classify"):
                cleaned_text = clean_text(news_text)
                vectorized_text = vectorizer.transform([cleaned_text])
                prediction = model.predict(vectorized_text)
                st.success(f"Text Categorized as: {prediction[0]}")

    if page_selection == "Solution Overview":
        st.title("Solution Overview")

        st.subheader("1. Importing Packages")
        st.write("Begin by importing the necessary Python packages, including libraries such as NumPy, Pandas, and scikit-learn, to facilitate data manipulation, analysis, and modeling.")
    
        st.subheader("2. Loading Data")
        st.write("Load the dataset into your working environment, ensuring that it is accessible and ready for analysis. This step may involve reading data from files, databases, or APIs.")

        st.subheader("3. Exploratory Data Analysis (EDA)")
        st.write("Perform a comprehensive EDA to gain insights into the dataset's characteristics.")
        st.write("Visualize and summarize key statistics, identify patterns, and detect outliers to inform subsequent preprocessing steps.")

        st.subheader("4. Preprocessing")
        st.write("Clean and prepare the data for model training.")
        st.write("This may include handling missing values, encoding categorical variables, scaling numerical features, and other tasks to ensure the data is suitable for machine learning algorithms.")

        st.subheader("5. Model Training")
        st.write("Select and train machine learning models on the preprocessed data.")
        st.write("This step involves splitting the dataset into training and testing sets, choosing appropriate algorithms, and fine-tuning model parameters for optimal performance")

        st.subheader("6. Model Evaluation")
        st.write("Assess the trained models' performance using evaluation metrics such as accuracy, precision, recall, and F1 score. Utilize techniques like cross-validation to ensure robust evaluation and avoid overfitting.")

        st.subheader("7. Best Model Selection")
        st.write("Identify the model that performs best based on the evaluation results.")
        st.write("Consider factors like accuracy, interpretability, and computational efficiency when choosing the final model for deployment.")

        st.subheader("8. Best Model Explanation")
        st.write("Provide an explanation for why the selected model is considered the best.")
        st.write("This may involve interpreting feature importance, visualizing decision boundaries, or explaining how the model captures patterns in the data.")

        st.subheader("9. Model Performance")
        st.write("Evaluate the model performance on the test dataset.")
        
        st.code("""
# Evaluate models on test data
test_performance = {}
for model_name, model in models.items():
    y_test_pred = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_performance[model_name] = test_accuracy
    print(f"{model_name} Test Accuracy: {test_accuracy}")
    print(f"\\nClassification Report for {model_name}:\\n", classification_report(y_test, y_test_pred))

# Example output:
Logistic Regression Test Accuracy: 0.97

Classification Report for Logistic Regression:
                precision    recall  f1-score   support

     business       0.96      0.96      0.96       400
    education       0.99      0.98      0.99       400
entertainment       0.99      0.97      0.98       400
       sports       0.98      0.98      0.98       400
   technology       0.92      0.96      0.94       400

     accuracy                           0.97      2000
    macro avg       0.97      0.97      0.97      2000
 weighted avg       0.97      0.97      0.97      2000
        """)
        
        st.subheader("10. Insights & Recommendations")
        st.write("Insights:")
        st.write("""
- Logistic Regression, SVM, Naive Bayes, and Neural Networks all showed high accuracy on both validation and test sets, indicating their robustness and generalizability.
- Random Forest also performed well, though slightly behind the aforementioned models.
- Decision Trees provided reasonable accuracy but were outperformed by ensemble methods like Random Forest and more complex models like SVM and Neural Networks.
        """)

        st.write("Recommendations:")
        st.write("""
- Neural Networks and Naive Bayes are recommended for their high accuracy and consistent performance across all categories.
- SVM and Logistic Regression are also strong contenders due to their robustness and generalizability.
- Random Forest is a good option for its balance between simplicity and performance.
- Further improvements can be made by hyperparameter tuning for SVM and Neural Network models.
- Collect more data to enhance model robustness, especially for underrepresented categories.
- Implement cross-validation to ensure the model's stability across different data splits.
        """)

    if page_selection == "About":
        st.title("About")
        st.markdown("""
        Insights Lab Consulting specializes in transforming complex data into actionable insights that drive informed decision-making.
        Our team leverages advanced analytics, machine learning algorithms, and domain expertise to extract valuable insights from data.
        Whether it's predictive modeling, trend analysis, or optimization strategies, Insight Lab is committed to delivering tangible outcomes that empower our clients to succeed.
        """)

    if page_selection == "Contact Us":
        st.title("Contact Us")
        st.markdown("""
        - Email: contact@insightslab.com
        - Phone: +1 234 567 890
        - Address: 123 Data Street, Data City, DC 45678
        """)

    if page_selection == "For The Tech Geeks":
        st.title("For The Tech Geeks")
        st.markdown("""
        This section provides technical details for developers and data scientists interested in the implementation of the news categorization system.
        We utilized the following machine learning models and techniques:
        - Logistic Regression
        - Support Vector Machines (SVM)
        - Naive Bayes
        - Random Forest
        - Neural Networks
        The models were trained using TF-IDF vectorized text data. 
        For more information, refer to our GitHub repository.
        """)

    if page_selection == "FAQ":
        st.title("FAQ")
        st.markdown("""
        **Q: How does the news categorization system work?**
        A: The system uses machine learning models to classify news articles into predefined categories based on their content.

        **Q: Which machine learning models are used?**
        A: We use a combination of Logistic Regression, SVM, Naive Bayes, Random Forest, and Neural Networks.

        **Q: How accurate are the models?**
        A: The models have been evaluated on a test dataset with high accuracy scores, particularly for Neural Networks and Naive Bayes.
        """)
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()  