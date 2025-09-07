#!/bin/sh
set -e

# Gunakan variabel dari environment, atau fallback ke model default
MODEL_LIST="${OLLAMA_MODELS:-mistral:7b-instruct-q4}"

echo "Menjalankan ollama serve di background..."
ollama serve &

# Simpan PID proses ollama serve
OLLAMA_PID=$!

echo "Menunggu Ollama service tersedia..."

# Tunggu sampai ollama list berhasil dijalankan
until ollama list > /dev/null 2>&1; do
  echo "Ollama belum siap, tunggu sebentar lagi..."
  sleep 2
done

echo "Ollama sudah tersedia."

# Loop untuk semua model yang ditentukan
for MODEL_NAME in $MODEL_LIST; do
  if ! ollama list | grep -q "$MODEL_NAME"; then
    echo "Model $MODEL_NAME belum ditemukan. Menjalankan pull..."
    ollama pull "$MODEL_NAME"
  else
    echo "Model $MODEL_NAME sudah tersedia."
  fi
done

echo "Semua model siap. Menjaga container tetap hidup..."

# Tunggu proses ollama serve agar container tetap berjalan
wait $OLLAMA_PID