# Invoice-Similarity-Finder

Step 1: Document Representation

Function: extract_text_from_pdf(pdf_path)
Opens a PDF file and extracts text from each page.
Function: pdf_to_image(pdf_path)
Opens a PDF file and converts each page to an image using PyMuPDF.


Step 2: Feature Extraction

Function: extract_features(text)
Uses TF-IDF to vectorize the extracted text.
Extracts invoice numbers, dates, and amounts using regular expressions.


Step 3: Similarity Calculation

Function: calculate_similarity(features1, features2)
Computes cosine similarity for TF-IDF vectors.
Computes Jaccard similarity for invoice numbers, dates, and amounts.
Function: calculate_image_similarity(image1, image2)
Converts images to grayscale and resizes them.
Computes SSIM between the two images.


Step 4: Database Integration

Variable: invoices_db
A list to store tuples of PDF paths, extracted features, and images.
Function: add_invoice_to_db(pdf_path)
Extracts text, features, and images from a PDF and adds them to invoices_db.
Function: find_most_similar_invoice(pdf_path)
Extracts text, features, and images from a new PDF.
Compares the new invoice to all stored invoices to find the most similar one.


Step 5: Output

The script will print the path of the most similar invoice and its similarity score.
