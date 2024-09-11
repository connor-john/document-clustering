import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import PyPDF2
import numpy as np

# Download necessary NLTK data
nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("stopwords")


def parse_data(directory):
    documents = []
    filenames = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                documents.append(file.read())
                filenames.append(filename)
        elif filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
            if text:
                documents.append(text)
                filenames.append(filename)
    return documents, filenames


def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def chunk_documents(documents, chunk_size=200):
    chunks = []
    chunk_info = []
    for doc_idx, doc in enumerate(documents):
        sentences = sent_tokenize(doc)
        current_chunk = []
        current_chunk_size = 0
        for sent in sentences:
            words = word_tokenize(sent)
            if current_chunk_size + len(words) > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    chunk_info.append(doc_idx)
                current_chunk = words
                current_chunk_size = len(words)
            else:
                current_chunk.extend(words)
                current_chunk_size += len(words)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            chunk_info.append(doc_idx)
    return chunks, chunk_info


def preprocess(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    return " ".join(
        [word for word in words if word.isalnum() and word not in stop_words]
    )


def vectorize(chunks):
    vectorizer = TfidfVectorizer(preprocessor=preprocess)
    vectors = vectorizer.fit_transform(chunks)
    return vectors, vectorizer


def k_means(vectors, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(vectors)
    return kmeans.labels_


def visualize(vectors, chunk_info, filenames):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors.toarray())

    plt.figure(figsize=(20, 15))

    # Create a color map for documents
    unique_docs = list(set(chunk_info))
    color_map = plt.cm.get_cmap("tab20")  # You can change 'tab20' to other colormaps
    doc_colors = [color_map(i / len(unique_docs)) for i in range(len(unique_docs))]

    # Plot each point
    for i, (x, y) in enumerate(coords):
        doc_idx = chunk_info[i]
        plt.scatter(x, y, c=[doc_colors[unique_docs.index(doc_idx)]], s=50, alpha=0.7)

    # Create a custom legend for documents
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=filenames[doc_idx],
            markerfacecolor=doc_colors[i],
            markersize=10,
        )
        for i, doc_idx in enumerate(unique_docs)
    ]

    # Add legend
    plt.legend(
        handles=legend_elements,
        title="Documents",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    plt.title("Document Content Clustering Visualization")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.tight_layout()
    plt.show()


def main():
    directory = "./data"
    documents, filenames = parse_data(directory)

    if not documents:
        print("No documents found in the specified directory.")
        return

    print(f"Found {len(documents)} documents.")

    chunks, chunk_info = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks across all documents.")

    # TODO: Kmeans

    vectors, vectorizer = vectorize(chunks)

    visualize(vectors, chunk_info, filenames)


if __name__ == "__main__":
    main()
