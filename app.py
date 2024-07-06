import streamlit as st
st.set_page_config(layout="wide")
import pickle
import pandas as pd
from cleantext import clean
from collections import Counter
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image

# Download NLTK stopwords
nltk.download('stopwords')

# Function to clean text
def clean_text_for_ml(text):
    return clean(text,
                 fix_unicode=True,       # fix various unicode errors
                 to_ascii=True,          # transliterate to closest ASCII representation
                 lower=True,             # lowercase text
                 no_line_breaks=True,    # remove line breaks
                 no_urls=True,           # remove URLs
                 no_emails=True,         # remove email addresses
                 no_phone_numbers=True,  # remove phone numbers
                 no_numbers=True,        # remove all numbers
                 no_digits=True,         # remove all digits
                 no_currency_symbols=True,  # remove currency symbols
                 no_punct=True,          # remove punctuation
                 replace_with_punct="",  # replace punctuation with nothing
                 lang="en"               # set the language to English
                 )

# Function to plot most common words
def plot_most_common_words(text_series, num=30):
    words = ' '.join(text_series).lower().split()
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    freq_dist = nltk.FreqDist(filtered_words)
    most_common_words = freq_dist.most_common(num)
    words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])

    fig, ax = plt.subplots(figsize=(10, 5))
    words_df.plot(kind='bar', x='Word', y='Frequency', ax=ax, legend=False)
    plt.xticks(rotation=45)
    plt.title(f'Top {num} Most Common Words')
    st.pyplot(fig)

# Function to create a word cloud
def plot_wordcloud(text_series):
    text = ' '.join(text_series).lower()
    filtered_text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Function to resize images
def resize_image(image_path, size=(300, 300)):
    img = Image.open(image_path)
    img = img.resize(size)
    return img

# Load the models
model_files = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    'Decision Tree': 'decision_tree_model.pkl',
    'Gradient Boosting': 'gradient_boosting_model.pkl',
    'Support Vector Machine': 'support_vector_machine_model.pkl'
}

models = {}
for name, file in model_files.items():
    with open(file, 'rb') as f:
        models[name] = pickle.load(f)

# Load the CSV file
file_path = 'train.csv'
train_data = pd.read_csv(file_path)

# Streamlit app
st.title("News Category Prediction")

# Sidebar for navigation
st.sidebar.image("images/vhp_img5720.jpg", use_column_width=True)
option = st.sidebar.selectbox(
    'Select a page:',
    ['Predictions', 'EDA', 'About Us', 'Our Models']
)

if option == 'Predictions':
    # Text input
    input_text = st.text_area("Enter the news headline:")

    if st.button("Predict"):
        if input_text:
            # Clean the input text
            cleaned_text = clean_text_for_ml(input_text)

            # Prepare data for prediction
            input_df = pd.DataFrame([cleaned_text], columns=['cleaned_text'])

            # Predict using each model
            predictions = {name: model.predict(input_df['cleaned_text'])[0] for name, model in models.items()}

            # Count the votes for each prediction
            prediction_votes = Counter(predictions.values())
            final_prediction = prediction_votes.most_common(1)[0][0]

            # Display individual predictions with HTML and CSS styling
            st.markdown("## Individual Model Predictions:")
            for name, prediction in predictions.items():
                st.markdown(f"""
                <div style="padding: 10px; border: 1px solid #ccc; margin: 10px 0; border-radius: 5px;">
                    <strong>{name}:</strong> {prediction}
                </div>
                """, unsafe_allow_html=True)

            # Display the final prediction based on majority vote
            st.markdown(f"""
            <div style="padding: 20px; border: 2px solid #28a745; background-color: #dff0d8; border-radius: 5px; margin: 20px 0;">
                <strong>Final Prediction based on majority vote:</strong> {final_prediction}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write("Please enter a news headline to predict.")

elif option == 'EDA':
    st.write("## Exploratory Data Analysis (EDA)")

    # Create columns for side-by-side plots
    col1, col2 = st.columns(2)

    # Category distribution
    with col1:
        st.write("### Category Distribution")
        category_counts = train_data['category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        fig_category = px.bar(category_counts, x='Category', y='Count', title='Number of Articles per Category')
        st.plotly_chart(fig_category)

    # Word count distribution in headlines
    with col2:
        st.write("### Word Count Distribution in Headlines")
        train_data['headline_word_count'] = train_data['headlines'].apply(lambda x: len(x.split()))
        fig_word_count = px.histogram(train_data, x='headline_word_count', nbins=20, title='Word Count Distribution in Headlines')
        st.plotly_chart(fig_word_count)

    # Create columns for most common words and word cloud
    col3, col4 = st.columns(2)

    with col3:
        st.write("### Most Common Words in Headlines")
        plot_most_common_words(train_data['headlines'])

    with col4:
        st.write("### Word Cloud of Headlines")
        plot_wordcloud(train_data['headlines'])

elif option == 'About Us':
    # Card-like container for the About Us section
    st.markdown("""
    <div style="padding: 20px; border: 1px solid #ccc; border-radius: 10px; margin: 10px; background-color: #f9f9f9;">
        <h2>About Us</h2>
        <p>This application was developed to demonstrate the use of multiple machine learning models to predict news categories.
        It cleans the input text and makes predictions using various models, displaying both individual model predictions and a
        final prediction based on majority voting.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("### Meet Our Team")

    # Team member images and descriptions
    team_members = [
        {"name": "Asanda Gambu", "role": "Data Scientist", "image": "images/team_member_1.jpg"},
        {"name": "Makhutjo Lehutjo", "role": "Github Manager", "image": "images/team_member_2.jpg"},
        {"name": "Khululiwe Hlongwane", "role": "Project Manager", "image": "images/team_member_3.jpg"},
    ]

    col1, col2, col3 = st.columns(3)

    for idx, member in enumerate(team_members):
        with [col1, col2, col3][idx]:
            img = resize_image(member['image'])
            st.image(img, use_column_width=True)
            st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <h3>{member['name']}</h3>
                <p>{member['role']}</p>
            </div>
            """, unsafe_allow_html=True)

elif option == 'Our Models':
    st.write("## Our Models")

    models_info = [
        {
            "title": "Logistic Regression",
            "description": """
            Logistic Regression is a linear model commonly used for classification tasks. It works well when the relationship between 
            the features and the target variable is approximately linear. We chose Logistic Regression for its simplicity and efficiency.
            """
        },
        {
            "title": "Decision Tree",
            "description": """
            Decision Tree is a non-linear model that splits the data into subsets based on the feature values, making it easy to understand
            and interpret. We chose Decision Tree for its ability to handle non-linear relationships and its interpretability.
            """
        },
        {
            "title": "Gradient Boosting",
            "description": """
            Gradient Boosting is an ensemble model that builds trees sequentially, each one correcting the errors of the previous one.
            It is highly effective for improving predictive accuracy. We chose Gradient Boosting for its ability to enhance performance 
            through boosting.
            """
        },
        {
            "title": "Support Vector Machine",
            "description": """
            Support Vector Machine (SVM) is a powerful classification model that finds the hyperplane that best separates the classes.
            It is effective for high-dimensional spaces and when the number of dimensions is greater than the number of samples. 
            We chose SVM for its effectiveness in complex classification tasks.
            """
        }
    ]

    for model in models_info:
        st.markdown(f"""
        <div style="padding: 20px; border: 1px solid #ccc; border-radius: 10px; margin: 10px; background-color: #f9f9f9;">
            <h3>{model['title']}</h3>
            <p>{model['description']}</p>
        </div>
        """, unsafe_allow_html=True)
