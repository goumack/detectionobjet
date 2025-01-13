import streamlit as st
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import requests  # Importer pour la gestion des requêtes HTTP

# Chargez le modèle et le processeur
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

def get_image_from_url(url):
    """Télécharge l'image à partir d'une URL"""
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

def detect_objects(image):
    """Détecte les objets dans l'image"""
    # Préparez l'image
    inputs = processor(images=image, return_tensors="pt")

    # Obtenez les prédictions
    outputs = model(**inputs)
    logits = outputs.logits
    boxes = outputs.pred_boxes

    # Appliquez softmax pour les probabilités
    probs = torch.nn.functional.softmax(logits[0], dim=-1)
    scores, labels = probs[..., :-1].max(dim=-1)

    # Filtrez avec un seuil
    threshold = 0.5
    keep = scores > threshold

    # Ajustez les dimensions des boîtes
    width, height = image.size
    boxes = boxes.squeeze(0)
    boxes = boxes * torch.tensor([width, height, width, height])

    # Filtrez les prédictions
    filtered_boxes = boxes[keep]
    filtered_labels = labels[keep]
    filtered_scores = scores[keep]

    # Comptage des entités pour chaque classe
    label_counts = Counter(filtered_labels.tolist())

    return filtered_boxes, filtered_labels, filtered_scores, label_counts

def plot_image_with_boxes(image, filtered_boxes, filtered_labels, filtered_scores):
    """Affiche l'image avec les boîtes de détection"""
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
        x, y, width, height = box.tolist()
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y, f"{model.config.id2label[label.item()]} ({score:.2f})", color="red", fontsize=12)

    st.pyplot(fig)

def main():
    st.title("Détection d'objets ACCEL-2025")
    
    # Initialiser la variable 'image' à None pour éviter l'erreur UnboundLocalError
    image = None
    
    # Choisir entre télécharger une image locale ou entrer une URL
    option = st.radio("Choisissez l'entrée de l'image", ("Télécharger une image locale", "Entrer une URL"))
    
    if option == "Télécharger une image locale":
        image_file = st.file_uploader("Choisissez une image", type=["jpg", "png", "jpeg"])
        
        if image_file is not None:
            image = Image.open(image_file).convert("RGB")
            
    elif option == "Entrer une URL":
        url = st.text_input("Entrez l'URL de l'image")  # Champ pour entrer l'URL
        
        if url:
            try:
                image = get_image_from_url(url)
            except Exception as e:
                st.error(f"Erreur lors du téléchargement de l'image : {e}")
                return

    # Vérifiez que l'image a bien été chargée
    if image is not None:
        # Détection des objets
        filtered_boxes, filtered_labels, filtered_scores, label_counts = detect_objects(image)
        
        # Affichage des résultats de comptage
        st.write("Nombre d'objets détectés pour chaque classe :")
        for label, count in label_counts.items():
            class_name = model.config.id2label[label]
            st.write(f"{class_name}: {count}")
        
        # Affichage de l'image avec les boîtes
        plot_image_with_boxes(image, filtered_boxes, filtered_labels, filtered_scores)

if __name__ == "__main__":
    main()
