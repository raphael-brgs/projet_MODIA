# Utiliser une image de base Python
FROM python:3.9

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Mettre à jour pip
RUN pip install --no-cache-dir --upgrade pip

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances Python spécifiées dans requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Créer les dossiers pickles/ et pretrained-model/ s'ils n'existent pas déjà
RUN mkdir -p pickles pretrained-model

# Télécharger les données depuis Google Drive
RUN gdown 'https://drive.google.com/uc?export=download&id=1y8uKI9ocGnCHJRJiflbHMxQt8Tvb-Hfg' -O pickles/vectorizer.pickle \
    && gdown 'https://drive.google.com/uc?export=download&id=1S1_UDg4KCLr6nxIYE98_jfOmi6dOeSEe' -O pretrained-model/pytorch_model.bin

# Copier le reste du code source dans le conteneur
COPY . .
