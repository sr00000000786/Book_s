import streamlit as st
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image
from io import BytesIO
import requests
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load data
books = pd.read_csv("Books.csv", encoding='ISO-8859-1')
ratings = pd.read_csv("Ratings.csv", encoding='ISO-8859-1')
users = pd.read_csv("Users.csv", encoding='ISO-8859-1')

# Drop rows with null values from the 'books' DataFrame
books.dropna(inplace=True)
# Drop rows with null values from the 'users' DataFrame
users.dropna(inplace=True)
# Drop columns 'Location' and 'Age' from the 'users' DataFrame
users.drop(columns=['Location', 'Age'], inplace=True)

# Merge dataframes
df1 = pd.merge(ratings, users, on='User-ID')
df3 = pd.merge(df1, books, on='ISBN')

# App title
st.title("Book Recommendation App")

# User input
recommendation_type = st.selectbox("Select Recommendation Type", ["User Based", "Content Based"])

if recommendation_type == "User Based":
    # User input
    target_user = st.number_input("Enter a User ID:", min_value=1, max_value=df3['User-ID'].max())

    # Assuming you have loaded your dataframe as 'df3'
    user_item_matrix = coo_matrix((df3['Book-Rating'],
                               (df3['User-ID'], df3['ISBN'].astype('category').cat.codes)))
    user_item_matrix = user_item_matrix.tocsr()

    n_components = 50
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(user_item_matrix)

    U = svd.transform(user_item_matrix)
    Vt = svd.components_
    item_embeddings = Vt.T

    num_trees = 50
    embedding_size = item_embeddings.shape[1]
    annoy_index = AnnoyIndex(embedding_size, metric='angular')

    for i in range(len(item_embeddings)):
        annoy_index.add_item(i, item_embeddings[i])

    annoy_index.build(num_trees)

    def recommend_books(user_id, num_recommendations=5):
        user_vector = U[user_id]
        approximate_neighbors = annoy_index.get_nns_by_vector(user_vector, num_recommendations, search_k=-1)
        recommended_books = df3.loc[approximate_neighbors, ['Book-Title', 'Image-URL-S']]
        return recommended_books

    if st.button("Get Recommendations"):
        recommended_books = recommend_books(target_user)

        st.subheader("Recommended Books:")

        # Display images in a grid
        cols = st.columns(3)  # Display 3 images per row
        col_counter = 0
        for index, row in recommended_books.iterrows():
            if col_counter >= 3:
                col_counter = 0

            with cols[col_counter]:
                st.image(row['Image-URL-S'], caption=row['Book-Title'], use_column_width=True)
                st.write("Title:", row['Book-Title'])
                col_counter += 1
else:
         # Sample a subset of the dataset for testing and development
     sample_size = 10000  # Adjust the sample size as needed
     sample_df = df3.sample(n=sample_size, random_state=42)

     # Split the sample dataset into a training set and a testing set
     train_df, test_df = train_test_split(sample_df, test_size=0.2, random_state=42)

     # Preprocess text data for content-based filtering
     tfidf_vectorizer = TfidfVectorizer(stop_words='english')
     item_features_train = tfidf_vectorizer.fit_transform(train_df['Book-Title'] + ' ' + train_df['Book-Author'])

     # Calculate cosine similarity between items in the training set
     item_similarity_train = linear_kernel(item_features_train, item_features_train)

     # Create a mapping between sample indices and full dataset indices
     sample_indices = train_df.index
     full_indices = df3[df3.index.isin(sample_indices)].index

     def recommend_books_content_based(book_title, num_recommendations=5):
         book_indices = sample_df[sample_df['Book-Title'] == book_title].index
         recommended_books = set()  # Using a set to avoid duplicates
         for book_index in book_indices:
             if book_index in full_indices:
                 full_book_index = full_indices.get_loc(book_index)
                 similar_books_indices = np.argsort(item_similarity_train[full_book_index])[::-1][1:num_recommendations+1]
                 recommended_books.update(df3.loc[similar_books_indices, 'Book-Title'])
         return list(recommended_books)
     
     target_book = st.text_input("Enter a Book Title:", "The Da Vinci Code")

     if st.button("Get Recommendations (Content-Based)"):
         recommended_books_content_based = recommend_books_content_based(target_book)

         st.subheader("Recommended Book Titles and Images (Content-Based):")
    
         if len(recommended_books_content_based) == 0:
             st.write("No recommendations found.")
         else:
             for book_title in recommended_books_content_based:
                 book_info = df3[df3['Book-Title'] == book_title].iloc[0]
                 st.write(book_title)
                 st.image(book_info['Image-URL-L'], use_column_width=True)