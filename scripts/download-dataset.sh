%%writefile /content/drive/MyDrive/Deep_learning/DL/download-dataset.sh
#!/bin/bash

NUM_PROC=6
SPLIT=$1
N=$2

# Définir le dossier cible pour l'enregistrement des fichiers
TARGET_DIR="/content/drive/MyDrive/Deep_learning/DL/data/train/images"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

download_check_and_extract() {
  local i=$1
  images_file_name="images_$i.tar"
  images_md5_file_name="md5.images_$i.txt"
  images_tar_url="https://s3.amazonaws.com/google-landmark/$SPLIT/$images_file_name"
  images_md5_url="https://s3.amazonaws.com/google-landmark/md5sum/$SPLIT/$images_md5_file_name"

  # Vérifier si le fichier TAR existe déjà et est extrait
  extracted_folder="${images_file_name%.tar}"
  if [[ -f "$images_file_name" || -d "$extracted_folder" ]]; then
    echo "✅ $images_file_name déjà téléchargé et/ou extrait, passage au suivant."
    return
  fi

  echo "📥 Téléchargement de $images_file_name et de son md5sum..."
  curl -Os "$images_tar_url"
  curl -Os "$images_md5_url"

  if [[ "$OSTYPE" == "linux-gnu" || "$OSTYPE" == "linux" ]]; then
    images_md5="$(md5sum "$images_file_name" | cut -d" " -f1)"
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    images_md5="$(md5 -r "$images_file_name" | cut -d" " -f1)"
  fi

  md5_2="$(cut -d" " -f1 < "$images_md5_file_name")"

  if [[ "$images_md5" != "" && "$images_md5" == "$md5_2" ]]; then
    echo "🗄️ Extraction de $images_file_name..."
    tar -xf "$images_file_name"
    echo "✅ $images_file_name extrait dans $TARGET_DIR !"
  else
    echo "❌ MD5 checksum pour $images_file_name ne correspond pas à $images_md5_file_name, fichier corrompu."
  fi
}

for i in $(seq 0 $NUM_PROC $N); do
  upper=$(expr $i + $NUM_PROC - 1)
  limit=$((upper>N ? N : upper))
  for j in $(seq -f "%03g" $i $limit); do 
    download_check_and_extract "$j" & 
  done
  wait
done
