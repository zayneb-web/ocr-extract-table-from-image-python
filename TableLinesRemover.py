import cv2
import numpy as np
import os

class TableLinesRemover:

    def __init__(self, image):
        if image is None:
            raise Exception("L'image d'entrée est None")
        self.image = image
        
        # Créer le dossier pour les images de processus s'il n'existe pas
        os.makedirs("./process_images/table_lines_remover2", exist_ok=True)

    def execute(self):
        try:
            self.grayscale_image()
            self.store_process_image("0_grayscaled.jpg", self.grey)
            self.threshold_image()
            self.store_process_image("1_thresholded.jpg", self.thresholded_image)
            self.invert_image()
            self.store_process_image("2_inverted.jpg", self.inverted_image)
            self.erode_vertical_lines()
            self.store_process_image("3_erode_vertical_lines.jpg", self.vertical_lines_eroded_image)
            self.erode_horizontal_lines()
            self.store_process_image("4_erode_horizontal_lines.jpg", self.horizontal_lines_eroded_image)
            self.combine_eroded_images()
            self.store_process_image("5_combined_eroded_images.jpg", self.combined_image)
            self.dilate_combined_image_to_make_lines_thicker()
            self.store_process_image("6_dilated_combined_image.jpg", self.combined_image_dilated)
            self.subtract_combined_and_dilated_image_from_original_image()
            self.store_process_image("7_image_without_lines.jpg", self.image_without_lines)
            self.remove_noise_with_erode_and_dilate()
            self.store_process_image("8_image_without_lines_noise_removed.jpg", self.image_without_lines_noise_removed)
            cv2.imwrite("debug_vertical_lines_eroded.jpg", self.vertical_lines_eroded_image)
            return self.image_without_lines_noise_removed
        except Exception as e:
            print(f"Une erreur s'est produite lors du traitement de l'image : {str(e)}")
            return None

    def grayscale_image(self):
        self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if self.grey is None:
            raise Exception("Échec de la conversion en niveaux de gris")

    def threshold_image(self):
        ret, self.thresholded_image = cv2.threshold(self.grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.thresholded_image is None:
            raise Exception("Échec du seuillage de l'image")
        
        white_pixels = cv2.countNonZero(self.thresholded_image)
        black_pixels = self.thresholded_image.size - white_pixels

        if white_pixels > black_pixels:
            self.thresholded_image = cv2.bitwise_not(self.thresholded_image)

    def invert_image(self):
        if self.thresholded_image is None:
            raise Exception("Image seuillée non disponible pour l'inversion")
        self.inverted_image = self.thresholded_image

    def erode_vertical_lines(self):
        hor = np.array([[1,1,1,1,1,1]])
        self.vertical_lines_eroded_image = cv2.erode(self.inverted_image, hor, iterations=5)
        self.vertical_lines_eroded_image = cv2.dilate(self.vertical_lines_eroded_image, hor, iterations=5)

    def erode_horizontal_lines(self):
        ver = np.array([[1],
               [1],
               [1],
               [1],
               [1],
               [1],
               [1]])
        self.horizontal_lines_eroded_image = cv2.erode(self.inverted_image, ver, iterations=10)
        self.horizontal_lines_eroded_image = cv2.dilate(self.horizontal_lines_eroded_image, ver, iterations=10)

    def combine_eroded_images(self):
        self.combined_image = cv2.add(self.vertical_lines_eroded_image, self.horizontal_lines_eroded_image)

    def dilate_combined_image_to_make_lines_thicker(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.combined_image_dilated = cv2.dilate(self.combined_image, kernel, iterations=5)

    def subtract_combined_and_dilated_image_from_original_image(self):
        self.image_without_lines = cv2.subtract(self.inverted_image, self.combined_image_dilated)

    def remove_noise_with_erode_and_dilate(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.image_without_lines_noise_removed = cv2.erode(self.image_without_lines, kernel, iterations=1)
        self.image_without_lines_noise_removed = cv2.dilate(self.image_without_lines_noise_removed, kernel, iterations=1)

    def store_process_image(self, file_name, image):
        if image is None:
            print(f"Warning: Tentative de sauvegarde d'une image None ({file_name})")
            return
        try:
            path = "./process_images/table_lines_remover/" + file_name
            cv2.imwrite(path, image)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'image {file_name}: {str(e)}")