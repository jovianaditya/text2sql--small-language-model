#!/bin/sh
set -e

# Ambil env custom; jangan pakai OLLAMA_MODELS karena itu env bawaan Ollama untuk path folder model
RAW="${OLLAMA_PULL_MODELS:-snowflake-arctic-embed2:568m,qwen2.5:7b-instruct,qwen2.5:3b-instruct,qwen2.5:1.5b-instruct}"

# Normalisasi: hapus kutip jika ada, ganti pemisah koma/spasi ganda jadi newline
MODEL_LIST=$(printf '%s' "$RAW" \
  | tr -d '\r' \
  | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//" -e 's/[[:space:]]\+/,/g' \
  | tr ',' '\n')

echo "Menjalankan ollama serve di background..."
ollama serve &
OLLAMA_PID=$!

echo "Menunggu Ollama service tersedia..."
until ollama list >/dev/null 2>&1; do
  echo "Ollama belum siap, tunggu sebentar lagi..."
  sleep 2
done

echo "Ollama sudah tersedia."
echo "Daftar model (RAW): '$RAW'"
echo "$MODEL_LIST" | while IFS= read -r MODEL_NAME; do
  [ -z "$MODEL_NAME" ] && continue
  echo " - $MODEL_NAME"
done

pull_with_retry() {
  name="$1"
  tries=3
  i=1

  while [ "$i" -le "$tries" ]; do
    echo "Pull $name (attempt $i/$tries)..."

    if ollama pull "$name"; then
      return 0
    fi

    echo "Pull $name gagal. Retry 5s..."
    sleep 5
    i=$((i+1))
  done

  echo "WARNING: Gagal pull $name setelah $tries percobaan. Lanjut ke model berikutnya."
  return 1
}

echo "$MODEL_LIST" | while IFS= read -r MODEL_NAME; do
  [ -z "$MODEL_NAME" ] && continue

  if ollama list | awk '{print $1}' | grep -Fqw "$MODEL_NAME"; then
    echo "Model $MODEL_NAME sudah tersedia."
  elif ollama list | awk '{print $1}' | grep -Fqw "${MODEL_NAME}:latest"; then
    echo "Model ${MODEL_NAME}:latest sudah tersedia."
  else
    pull_with_retry "$MODEL_NAME" || true
  fi
done

echo "Semua model dicek. Menjaga container tetap hidup..."
wait "$OLLAMA_PID"