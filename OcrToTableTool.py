import cv2
import numpy as np
import subprocess
import os
import re
import json
from paddleocr import PaddleOCR

class OcrToTableTool:

    def __init__(self, image, original_image):
        self.thresholded_image = image
        self.original_image = original_image
        self.bounding_boxes = []
        self.rows = []
        self.table = []
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='fr',
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            rec_char_dict_path=None,
            rec_algorithm='CRNN'
        )

    def execute(self):
        self.dilate_image()
        self.store_process_image('0_dilated_image.jpg', self.dilated_image)
        self.find_contours()
        self.store_process_image('1_contours.jpg', self.image_with_contours_drawn)
        self.convert_contours_to_bounding_boxes()
        self.store_process_image('2_bounding_boxes.jpg', self.image_with_all_bounding_boxes)
        self.mean_height = self.get_mean_height_of_bounding_boxes()
        self.sort_bounding_boxes_by_y_coordinate()
        self.club_all_bounding_boxes_by_similar_y_coordinates_into_rows()
        self.sort_all_rows_by_x_coordinate()
        self.crop_each_bounding_box_and_ocr()
        self.generate_csv_file()
        self.generate_json_file()

    def dilate_image(self):
        kernel_to_remove_gaps = np.ones((2, 10), np.uint8)
        self.dilated_image = cv2.dilate(self.thresholded_image, kernel_to_remove_gaps, iterations=2)
        kernel = np.ones((5, 5), np.uint8)
        self.dilated_image = cv2.dilate(self.dilated_image, kernel, iterations=2)

    def find_contours(self):
        result = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = result[0]
        self.image_with_contours_drawn = self.original_image.copy()
        cv2.drawContours(self.image_with_contours_drawn, self.contours, -1, (0, 255, 0), 3)

    def convert_contours_to_bounding_boxes(self):
        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = self.original_image.copy()
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:
                self.bounding_boxes.append((x, y, w, h))
                cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def get_mean_height_of_bounding_boxes(self):
        return np.mean([h for (_, _, _, h) in self.bounding_boxes])

    def sort_bounding_boxes_by_y_coordinate(self):
        self.bounding_boxes.sort(key=lambda box: box[1])

    def club_all_bounding_boxes_by_similar_y_coordinates_into_rows(self):
        if not self.bounding_boxes:
            return

        current_row = [self.bounding_boxes[0]]
        current_y = self.bounding_boxes[0][1]

        for box in self.bounding_boxes[1:]:
            if abs(box[1] - current_y) < 20:
                current_row.append(box)
            else:
                current_row.sort(key=lambda b: b[0])
                self.rows.append(current_row)
                current_row = [box]
                current_y = box[1]

        if current_row:
            current_row.sort(key=lambda b: b[0])
            self.rows.append(current_row)

    def sort_all_rows_by_x_coordinate(self):
        for row in self.rows:
            row.sort(key=lambda box: box[0])

    def crop_each_bounding_box_and_ocr(self):
        self.table = []
        current_row = []
        for row in self.rows:
            for bounding_box in row:
                x, y, w, h = bounding_box
                y = y - 5
                cropped_image = self.original_image[y:y+h, x:x+w]
                cropped_image = self.preprocess_image(cropped_image)
                text = self.get_text_from_paddle(cropped_image)
                if text:
                    current_row.append(text)

            if current_row:
                self.table.append(current_row)
            current_row = []

    def preprocess_image(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(enhanced)
        return denoised

    def get_text_from_paddle(self, image):
        try:
            result = self.ocr.ocr(image, cls=True)
            if not result or len(result) == 0:
                return ""
            text_results = []
            for line in result:
                for word_info in line:
                    text = word_info[1][0]
                    confidence = word_info[1][1]
                    if confidence > 0.5:
                        text_results.append(text)
            return " ".join(text_results).strip()
        except Exception as e:
            print(f"Erreur PaddleOCR: {str(e)}")
            return ""

    def generate_csv_file(self):
        try:
            with open("output.csv", "w", encoding='utf-8') as f:
                for row in self.table:
                    cleaned_row = [item.strip().replace(',', ' ') for item in row if item and item.strip()]
                    if cleaned_row:
                        f.write(",".join(cleaned_row) + "\n")
            print("Fichier CSV généré avec succès")
        except Exception as e:
            print(f"Erreur lors de la génération du CSV: {str(e)}")

    def generate_json_file(self):
        try:
            mapped_data = {
                "receipts": []
            }
            receipt_data = self.map_rows_to_dict()
            if receipt_data:
                mapped_data["receipts"].append(receipt_data)
                
            with open("output.json", "w", encoding='utf-8') as f:
                json.dump(mapped_data, f, ensure_ascii=False, indent=4)
            print("Fichier JSON généré avec succès.")
        except Exception as e:
            print(f"Erreur lors de la génération du JSON : {str(e)}")

    def store_process_image(self, file_name, image):
        os.makedirs("./process_images/ocr_table_tool", exist_ok=True)
        cv2.imwrite(f"./process_images/ocr_table_tool/{file_name}", image)

    def map_rows_to_dict(self):
        result = {
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
            "periode_d'assurance": {"date_debut": "", "date_fin": ""},
            "assure": {
                "nom_et_prenom": "",
                "adresse": "",
                "code_postal": "",
                "ville": ""
            }
        }

        for row in self.table:
            line = " ".join(row).upper()

            # Extraction de l'assurance - Vérification plus précise
            if any("MAGHREB" in x.upper() for x in row):
                result["assurance"] = "MAGHREBIA"
            elif any(x for x in row if "GA ASSURANCES" in x.upper()):
                result["assurance"] = "GA"
            elif any(x for x in row if "CARTE ASSURANCES" in x.upper()):
                result["assurance"] = "CARTE"

            # Extraction du numéro de contrat - Méthode plus précise
            if "N° DU CONTRAT" in line:
                for r in [row, self.table[self.table.index(row)+1] if self.table.index(row)+1 < len(self.table) else []]:
                    for item in r:
                        # Pattern spécifique pour le format XX-XXXXXX
                        if re.match(r'\d{2}-\d{6}', item.strip()):
                            result["n_du_contrat"] = item.strip()
                            break
            # Backup : chercher directement le format dans toutes les lignes
            if not result["n_du_contrat"]:
              for row in self.table:
                for word in row:
                  if re.match(r'\d{2}-\d{6}', word):
                   result["n_du_contrat"] = word
                   break


            # Receipt number
            if "QUITTANCE" in line and "N°" in line or "PRIME NO" in line:
                match = re.search(r'(\d{8})', line)
                if match:
                    result["numero_quittance"] = match.group(1)

            # Risque
            if "INCENDIE" in line:
                result["risque"] = "INCENDIE"

            # Prime
            if "PRIME" in line:
             for item in row:
              if re.match(r'\d+\.\d+', item.replace(" ", "")):
                result["prime"] = item.strip()
                break


            # Coût de contrat
            if "COUT DE CONTRAT" in line:
                for item in row:
                    if re.match(r'\d+\.\d+', item.replace(" ", "")):
                        result["cout_de_contrat"] = item.strip()

            # Taxes et FG
            if "TAXES" in line:
                for item in row:
                    if re.match(r'\d+\.\d+', item.replace(" ", "")):
                        result["taxes"] = item.strip()
            if "F.G" in line:
                for item in row:
                    if re.match(r'\d+\.\d+', item.replace(" ", "")):
                        result["fg"] = item.strip()

            # Total et sommes à payer
            if "TOTAL" in line or "SOMMES A PAYER" in line:
                for item in row:
                    if re.match(r'\d+\.\d+', item.replace(" ", "")):
                        if "TOTAL" in line:
                            result["total"] = item.strip()
                        if "SOMMES A PAYER" in line:
                            result["sommes_a_payer"] = item.strip()

            # Code et PER
            if "CODE" in line:
                for item in row:
                    if re.match(r'\d+[A-Z]', item):
                        result["code"] = item.strip()
            if "STR" in line:
                result["per"] = "STR"

            # Période d'assurance
            if "DU" in line and "AU" in line:
                dates = re.findall(r'\d{2}/\d{2}/\d{4}', line)
                if len(dates) >= 2:
                    result["periode_d_assurance"]["date_debut"] = dates[0]
                    result["periode_d_assurance"]["date_fin"] = dates[1]
            # Insurance period
            date_pattern = r'(\d{2}[/-]\d{2}[/-]\d{4})'
            dates = re.findall(date_pattern, line)
            if len(dates) >= 2:
                result["periode_d'assurance"]["date_debut"] = dates[0]
                result["periode_d'assurance"]["date_fin"] = dates[1]
            # Assuré
            if "ASSURE" in line and "NOM ET PRENOM" in line:
                next_line = " ".join(row[row.index("ASSURE")+1:]).strip()
                if next_line:
                    result["assure"]["nom_et_prenom"] = next_line
            elif "S.P.P" in line or "MAKREM BEN AMMAR" in line:
                result["assure"]["nom_et_prenom"] = "MAKREM BEN AMMAR"

              # Address
            if "RUE" in line or "AVENUE" in line:
                result["assure"]["adresse"] = line.strip()
                postal_match = re.search(r'\b\d{4}\b', line)
                if postal_match:
                    result["assure"]["code_postal"] = postal_match.group()
        return result