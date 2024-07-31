import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

# Step 1: Document Representation
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def pdf_to_image(pdf_path):
    document = fitz.open(pdf_path)
    images = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        images.append(img)
    return images

# Step 2: Feature Extraction
def extract_features(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()

    invoice_number = re.findall(r'Invoice Number: (\w+)', text)
    dates = re.findall(r'\d{2}/\d{2}/\d{4}', text)
    amounts = re.findall(r'\$\d+\.\d{2}', text)

    features = {
        'keywords': keywords,
        'invoice_number': invoice_number[0] if invoice_number else None,
        'dates': dates,
        'amounts': amounts
    }
    return features

# Step 3: Similarity Calculation
def calculate_similarity(features1, features2):
    vectorizer = TfidfVectorizer(stop_words='english')
    keywords1 = ' '.join(features1['keywords'])
    keywords2 = ' '.join(features2['keywords'])
    tfidf_matrix = vectorizer.fit_transform([keywords1, keywords2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    def jaccard_similarity(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    jaccard_sim_invoice_number = jaccard_similarity([features1['invoice_number']], [features2['invoice_number']])
    jaccard_sim_dates = jaccard_similarity(features1['dates'], features2['dates'])
    jaccard_sim_amounts = jaccard_similarity(features1['amounts'], features2['amounts'])

    combined_similarity = (cosine_sim + jaccard_sim_invoice_number + jaccard_sim_dates + jaccard_sim_amounts) / 4
    return combined_similarity

def calculate_image_similarity(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    image1 = cv2.resize(image1, (800, 800))
    image2 = cv2.resize(image2, (800, 800))

    score, _ = ssim(image1, image2, full=True)
    return score

# Step 4: Database Integration
invoices_db = []

def add_invoice_to_db(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    features = extract_features(text)
    images = pdf_to_image(pdf_path)
    invoices_db.append((pdf_path, features, images))

def find_most_similar_invoice(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    features = extract_features(text)
    images = pdf_to_image(pdf_path)

    max_similarity = 0
    most_similar_invoice = None

    for stored_pdf_path, stored_features, stored_images in invoices_db:
        feature_similarity = calculate_similarity(features, stored_features)
        image_similarity = calculate_image_similarity(images[0], stored_images[0])

        combined_similarity = (feature_similarity + image_similarity) / 2

        if combined_similarity > max_similarity:
            max_similarity = combined_similarity
            most_similar_invoice = stored_pdf_path

    return most_similar_invoice, max_similarity

# Step 5: Output
if __name__ == "__main__":
    invoice_paths = ["2024.03.15_0954.pdf", "Faller_8.pdf", "invoice_77073.pdf", "invoice_77098.pdf",
                     "invoice_102856.pdf", "invoice_102857.pdf"]

    for invoice_path in invoice_paths:
        add_invoice_to_db(invoice_path)

    new_invoice_path = "2024.03.15_1145.pdf"
    most_similar_invoice, similarity = find_most_similar_invoice(new_invoice_path)
    print(f"The most similar invoice is: {most_similar_invoice} with a similarity score of {similarity}")
