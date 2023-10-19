import torch
import streamlit as st
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pickle
import io

# Crear una instancia del modelo InceptionResnetV1 y MTCNN
model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(min_face_size=50, keep_all=False)

# Cargar la variable de características desde el archivo "caracteristicas.pkl"
with open("./caracteristicas.pkl", "rb") as f:
    caracteristicas = pickle.load(f)

# Funciones para calcular los embeddings (Tensores característicos de cada imagen) y la distancia euclidiana
def embedding(img_tensor):
    img_embedding = model(img_tensor.unsqueeze(0))
    return img_embedding

def Distancia(caracteristicas, embedding):
    distances = [(label, torch.dist(emb, embedding)) for label, emb in caracteristicas.items()]
    sorted_distances = sorted(distances, key=lambda x: x[1])
    return sorted_distances[0][0], sorted_distances[0][1].item()

# Crear una aplicación Streamlit
st.title("A que celebrity te pareces ?")

# Agregar un botón para cargar una imagen
uploaded_image = st.file_uploader("Cargar una imagen", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)

    # Verificar si la imagen está en formato PNG y convertir a JPG si es necesario
    if img.format == "PNG":
        jpg_io = io.BytesIO()  # Crear un objeto BytesIO para guardar la imagen en memoria
        img = img.convert("RGB")  # Convertir a modo RGB (requerido para guardar como JPG)
        img.save(jpg_io, format="JPEG")  # Guardar la imagen en el objeto BytesIO en formato JPG
        jpg_io.seek(0)  # Colocar el puntero de lectura al inicio del objeto BytesIO
        img = Image.open(jpg_io)  # Abrir la imagen en formato JPG desde el objeto BytesIO

    st.image(img, caption="Imagen cargada", use_column_width=True)

    if st.button("Reconocer Rostro"):
        if img is not None:
            # Procesar la imagen y realizar el reconocimiento facial
            img_tensor = mtcnn(img)
            if img_tensor is not None:
                embeddingNEW = embedding(img_tensor)
                result = Distancia(caracteristicas, embeddingNEW)
                st.write("Etiqueta predicha: ", result[0])
                st.write("Distancia euclidiana: ", result[1])
            else:
                st.write("No se encontró ningún rostro en la imagen.")
        else:
            st.write("Por favor, cargue una imagen válida.")


