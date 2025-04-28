import cv2
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import TableExtractor as te
import TableLinesRemover as tlr
import json
import re

# Initialize PaddleOCR
ocr = PaddleOCR(lang='en')  
# Process the image
path_to_image = "./image/HP0006.jpg"

# Extract table and correct perspective
table_extractor = te.TableExtractor(path_to_image)
perspective_corrected_image = table_extractor.execute()
cv2.imshow("perspective_corrected_image", perspective_corrected_image)

# Remove table lines
lines_remover = tlr.TableLinesRemover(perspective_corrected_image)
image_without_lines = lines_remover.execute()
cv2.imshow("image_without_lines", image_without_lines)

# Save the processed image temporarily
cv2.imwrite("temp_processed.jpg", image_without_lines)

# Perform OCR on the processed image
result = ocr.ocr("temp_processed.jpg", cls=False)

# Initialize the receipt data structure
receipt_data = {
    "receipts": [{
        "assurance": "",
        "n_du_contrat": "",
        "numero_quittance": "",
        "risque": "",
        "prime": "",
        "cout_de_contrat": "",
        "taxes": "",
        "fg": "",
        "total": "",
        "sommes_a_payer": "",
        "code": "",
        "per": "",
        "periode_d'assurance": {
            "date_debut": "",
            "date_fin": ""
        },
        "assure": {
            "nom_et_prenom": "",
            "adresse": "",
            "code_postal": "",
            "ville": ""
        }
    }]
}

boxes = []
texts = []
scores = []

print("\n=== Détection OCR ===")
# Process OCR results to extract information
for line in result:
    # PaddleOCR returns a list of lists, where each inner list contains [coordinates, [text, confidence]]
    if isinstance(line, list) and len(line) == 2:
        coordinates = line[0]
        text_info = line[1]
        if isinstance(text_info, list) and len(text_info) == 2:
            text = str(text_info[0]).strip()
            confidence = float(text_info[1])
            
            print(f"Texte détecté: '{text}' (confiance: {confidence:.2f})")
            
            # Add to drawing lists
            boxes.append(coordinates)
            texts.append(text)
            scores.append(confidence)
            
            # Extract numero quittance (format: 24406765)
            if text.isdigit() and len(text) >= 7:
                print(f"Numéro de quittance trouvé: {text}")
                receipt_data["receipts"][0]["numero_quittance"] = text
            
            # Extract risk type (INCENDIE)
            if text == "INCENDIE":
                print(f"Risque trouvé: {text}")
                receipt_data["receipts"][0]["risque"] = text
            
            # Extract dates (format: DD/MM/YYYY)
            date_pattern = r'\d{2}/\d{2}/\d{4}'
            dates = re.findall(date_pattern, text)
            if dates:
                print(f"Dates trouvées: {dates}")
                if len(dates) >= 2:
                    receipt_data["receipts"][0]["periode_d'assurance"]["date_debut"] = dates[0]
                    receipt_data["receipts"][0]["periode_d'assurance"]["date_fin"] = dates[1]
                elif len(dates) == 1:
                    if not receipt_data["receipts"][0]["periode_d'assurance"]["date_debut"]:
                        receipt_data["receipts"][0]["periode_d'assurance"]["date_debut"] = dates[0]
                    else:
                        receipt_data["receipts"][0]["periode_d'assurance"]["date_fin"] = dates[0]
            
            # Extract name (MAKREM BEN AMMAR)
            if "MAKREM" in text and "BEN" in text and "AMMAR" in text:
                print(f"Nom trouvé: {text}")
                receipt_data["receipts"][0]["assure"]["nom_et_prenom"] = "MAKREM BEN AMMAR"
            
            # Extract address (18RUEMAKTA)
            if any(c.isdigit() for c in text) and any(c.isalpha() for c in text) and len(text) > 5:
                print(f"Adresse trouvée: {text}")
                receipt_data["receipts"][0]["assure"]["adresse"] = text
            
            # Extract code postal (2081)
            if text.isdigit() and len(text) == 4:
                print(f"Code postal trouvé: {text}")
                receipt_data["receipts"][0]["assure"]["code_postal"] = text
            
            # Extract ville (ARIANA)
            if text == "ARIANA":
                print(f"Ville trouvée: {text}")
                receipt_data["receipts"][0]["assure"]["ville"] = text
            
            # Extract per (STR)
            if text == "STR":
                print(f"Per trouvé: {text}")
                receipt_data["receipts"][0]["per"] = text
            
            # Extract montants
            if re.match(r'^\d+\.?\d*$', text) or re.match(r'^\d+\s\d+$', text):
                print(f"Montant trouvé: {text}")
                if not receipt_data["receipts"][0]["prime"]:
                    receipt_data["receipts"][0]["prime"] = text
                elif not receipt_data["receipts"][0]["cout_de_contrat"]:
                    receipt_data["receipts"][0]["cout_de_contrat"] = text
                elif not receipt_data["receipts"][0]["taxes"]:
                    receipt_data["receipts"][0]["taxes"] = text
                elif not receipt_data["receipts"][0]["fg"]:
                    receipt_data["receipts"][0]["fg"] = text
                elif not receipt_data["receipts"][0]["total"]:
                    receipt_data["receipts"][0]["total"] = text

print("\n=== Résultats JSON ===")
print(json.dumps(receipt_data, indent=4, ensure_ascii=False))

# Save results to JSON file
with open('ocr_results.json', 'w', encoding='utf-8') as f:
    json.dump(receipt_data, f, ensure_ascii=False, indent=4)

# Draw results on the image
image = Image.open("temp_processed.jpg").convert('RGB')

# 📌 Définir la police
font_path = 'C:\\Windows\\Fonts\\arial.ttf'

# Draw OCR results
im_show = draw_ocr(image, boxes, texts, scores, font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')

# Show the final result
result_image = cv2.imread('result.jpg')
cv2.imshow("OCR Result", result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()