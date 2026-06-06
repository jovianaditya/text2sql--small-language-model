# Retrieval Policy

Dokumen ini menjelaskan proses retrieval di `t2sql` berdasarkan implementasi
aktual, terutama di:

- `data/python/mcp_client/t2sql/retrieval.py`
- `data/python/mcp_client/t2sql/helpers_main.py`
- `data/python/mcp_client/t2sql/chroma_embedding.py`
- `data/python/mcp_client/t2sql/config.py`

Fokus utama retrieval di sistem ini bukan menghasilkan SQL, tetapi membangun
context schema yang sempit, relevan, dan aman untuk generator SQL.

## 1. Output Retrieval Sebenarnya

Node retrieval mengisi:

- `state["context"]`
- `state["ranked_cols"]`
- `state["retrieval_snapshot"]`
- `state["retrieval_context_mode"]`

Isi `state["context"]` punya 3 blok:

```text
### ALLOWED IDENTIFIERS
- table_a: col1, col2

### CARDS
- table: table_a | column: col1 | desc: ...

### _INTERNAL_HINT
{"alias_map": {...}}
```

Artinya retrieval di sini berfungsi sebagai schema linker + context compressor.

## 2. Ringkasan Flow

Secara implementasi, `retrieve_node(state)` berjalan seperti ini:

```text
1. Bangun / reuse Chroma vector store
2. Ambil question dan tentukan qtype
3. Deteksi primary lookup table bila relevan
4. Pilih file sparse JSONL
5. Bangun sparse query dan jalankan BM25
6. Bangun dense probe dan jalankan Chroma query
7. Gabungkan dense candidate + sparse evidence
8. Hitung boosted score per candidate
9. Ambil top reranked candidates
10. Filter kolom yang relevan
11. Score tabel
12. Terapkan single-table lock atau multi-table expansion
13. Tambah bridge/FK path jika dibutuhkan
14. Bangun ALLOWED IDENTIFIERS, CARDS, alias_map
15. Simpan telemetry snapshot
```

## 3. Step-by-Step Sesuai Kode

### 3.1 Build vector store dulu

Retrieval selalu mulai dari:

```python
col = build_vector_store()
```

Perilakunya:

- membaca dense metadata dari `DENSE_METADATA_JSONL`
- membuat atau reuse collection Chroma
- auto-ingest bila collection kosong
- rebuild bila signature source dense berubah

Dense retrieval di sistem ini hanya bisa jalan kalau collection Chroma sudah
siap.

### 3.2 Query typing dan lookup focus

`retrieve_node()` mengambil:

- `q = state["question"]`
- `qtype = _get_qtype(state)`

Lalu untuk query lookup:

- `_detect_primary_entity_table(q)` mencari tabel yang disebut eksplisit
- `_wants_name_only_for_table(q, table)` mendeteksi apakah user kemungkinan
  meminta display name, bukan ID

Ini penting karena nanti mempengaruhi:

- `dense_k`
- auto-injection kolom name
- kemungkinan single-table lock

### 3.3 Pemilihan file sparse

Retrieval membaca `state["jsonl_paths"]`, lalu hanya memilih file yang
mengandung marker:

```text
RETRIEVAL_SPARSE_FILENAME_MARKER = "_sparse"
```

Kalau tidak ada, sistem fallback ke semua path yang diberikan.

Jadi sparse retrieval memang diarahkan ke metadata lexical JSONL, bukan ke
dense JSONL.

### 3.4 Sparse query build

Sparse query dibangun oleh:

```python
build_sparse_query_from_question(q)
```

Di dalamnya ada 2 tahap utama:

1. `_tokenize_bm25_query()`
2. `_phrase_to_column_map()` untuk phrase alias expansion berbasis schema

Normalisasi yang dilakukan:

- lowercase
- hilangkan thousand separator
- hilangkan possessive
- ganti non-alnum menjadi spasi
- stopword removal
- light stemming untuk plural
- pertahankan schema token penting agar tidak over-stem

Contoh:

```text
Question:
Show customers' names from USA

Sparse token kira-kira:
customer name usa
```

Kalau metadata punya sinonim multi-kata seperti:

```text
"list price" -> msrp
```

maka saat phrase itu muncul di question, token kolom asli bisa ikut
ditambahkan ke query sparse.

### 3.5 Sparse retrieval via BM25

Sparse retrieval dijalankan oleh:

```python
jsonl_sparse_search(sparse_query_clean, sparse_target_paths, max_hits=SPARSE_MAX_HITS)
```

Hal penting dari implementasinya:

- BM25 corpus dibangun dari JSONL sparse
- hasil token tiap dokumen diambil terutama dari `metadata.keywords`
- kalau `keywords` tidak ada, fallback ke `text`
- kalau tetap tidak ada, fallback ke gabungan table/column/description/aliases/roles/synonyms
- cache BM25 diikat ke hash isi file, bukan hanya nama file

ID dokumen sparse bisa berasal dari:

- `id`
- `field_urn`
- `urn`
- fallback format legacy

Sparse tidak langsung memberi context ke LLM. Sparse hanya memberi evidence
lexical untuk tahap scoring berikutnya.

### 3.6 Dense probe dan dense retrieval

Dense query dibangun oleh:

```python
query_text = build_dynamic_probe(q, qtype=qtype)
```

Probe ini tidak selalu sama dengan raw question. Ia bisa menambahkan hint
schema-driven seperti:

- label tabel yang disebut
- ID column atau display name column
- metric-like column
- temporal column
- geo hint seperti `country` atau `city`

Dense retrieval lalu dipanggil dengan:

```python
col.query(
    query_texts=[query_text],
    n_results=dense_k,
    include=["documents", "metadatas", "distances"],
)
```

Jumlah candidate dense:

```text
base_k = max(RETRIEVAL_BASE_K_FLOOR, TOP_K)
```

Default:

- `TOP_K = 5`
- `RETRIEVAL_BASE_K_FLOOR = 12`

Jadi normalnya dense mengambil minimal 12 candidate.

#### Kapan `dense_k` diperbesar?

Untuk lookup yang tampak meminta display name:

- ada `prefer_name_table`
- ada `preferred_name_col`
- sparse belum menemukan preferred name column

maka:

```text
dense_k = max(RETRIEVAL_DENSE_K_FLOOR, base_k)
```

Default `RETRIEVAL_DENSE_K_FLOOR = 28`.

Tujuannya: menaikkan recall dense ketika sistem curiga kolom display-name yang
penting belum muncul.

## 4. Reranking: Cara Kerjanya

Ini bagian inti.

Setelah dense result dibuka, setiap candidate dihitung score-nya dengan
komponen:

```text
base = -distance
boosted = base + sparse_bonus + priority_boost
```

Implementasinya ada di blok scoring dalam `retrieve_node()`.

### 4.1 Base score dari dense distance

Chroma memberi `distance`. Karena distance lebih kecil berarti lebih bagus,
kode membaliknya:

```python
base = -float(dist)
```

Jadi:

- distance `0.08` -> base `-0.08`
- distance `0.40` -> base `-0.40`

Candidate dengan distance lebih kecil akan punya base yang lebih tinggi
(kurang negatif).

### 4.2 Sparse bonus

Lalu sistem cek apakah dense candidate itu juga muncul di sparse hit:

```python
sparse_hit = _doc_matches_sparse(md, sparse_field_urns)
```

Kalau ya, ditambah:

```text
SPARSE_BONUS = 0.75
```

`_doc_matches_sparse()` mendukung beberapa format pencocokan:

- `table.column`
- `field_urn`
- `urn`
- suffix legacy `:table::column`

Jadi sparse berperan sebagai lexical confirmation terhadap candidate dense.

### 4.3 Priority boost berbasis intent + metadata

Setelah itu sistem memanggil:

```python
apply_column_priority_boost(q_l, table, col_name, roles, aliases)
```

Fungsi ini tidak hardcode nama tabel dataset. Ia memakai:

- `get_table_type(table)`
- role tags metadata
- pola nama kolom
- apakah tabel disebut di question
- apakah user minta `name` atau `id`

Contoh perilaku:

- bila user tampak meminta `name`, kolom nama bisa dapat bonus
- bila user tampak meminta `id`, kolom ID bisa dapat bonus
- pada dimension-like table, kolom `{entity}name` diberi prioritas tinggi
- contact name bisa dipenalti agar tidak mengalahkan entity name
- kolom ID bisa dipenalti ketika intent-nya jelas meminta nama

Beberapa default penting:

- `NAME_QUERY_BONUS = 0.3`
- `ID_QUERY_BONUS = 0.4`
- `COL_BOOST_DIM_NAME_EXPLICIT = 0.9`
- `COL_BOOST_DIM_NAME_DEFAULT = 0.6`
- `COL_BOOST_DIM_ID_PENALTY = -0.35`
- `COL_BOOST_DIM_CONTACT_PENALTY = -0.8`

## 5. Contoh Reranking dengan Angka

Misal user bertanya:

```text
Show customer names
```

Dense mengembalikan 3 candidate:

| Candidate | Distance | Base |
| --- | ---: | ---: |
| `customers.customerName` | 0.14 | -0.14 |
| `customers.contactLastName` | 0.11 | -0.11 |
| `customers.customerNumber` | 0.09 | -0.09 |

Sparse hit dari BM25:

- `customers.customerName`
- `customers.customerNumber`

Lalu anggap boost metadata seperti ini:

- `customerName`: `+0.3` name intent + `+0.9` explicit dimension-name
- `contactLastName`: `0.0` atau bahkan penalti contact
- `customerNumber`: `-0.35` karena ID saat user minta name

Maka boosted score kira-kira:

| Candidate | Base | Sparse | Priority | Boosted |
| --- | ---: | ---: | ---: | ---: |
| `customers.customerName` | -0.14 | +0.75 | +1.20 | 1.81 |
| `customers.customerNumber` | -0.09 | +0.75 | -0.35 | 0.31 |
| `customers.contactLastName` | -0.11 | +0.00 | -0.80 | -0.91 |

Hasilnya:

1. `customers.customerName`
2. `customers.customerNumber`
3. `customers.contactLastName`

Ini menunjukkan sesuatu yang penting:

- dense similarity saja tidak cukup
- sparse evidence bisa mengangkat candidate yang lexical-nya valid
- metadata boost menjaga kolom yang sesuai intent user tetap menang

## 6. Auto-Injection Preferred Name Column

Ada mekanisme khusus yang sangat sempit untuk lookup-name case.

Kondisi aktif:

- ada `prefer_name_table`
- ada `preferred_name_col`
- sparse sudah menemukan preferred name column
- dense belum menemukan kolom itu

Maka retrieval akan `col.get()` kolom itu dari Chroma lalu menyisipkannya ke
candidate list dengan skor:

```text
max_boosted_existing + RETRIEVAL_AUTO_INJECT_BONUS_DELTA
```

Default:

```text
RETRIEVAL_AUTO_INJECT_BONUS_DELTA = 0.05
```

Catatan penting:

- ini bukan hardcode nama kolom
- kolom yang di-inject tetap diambil dari metadata/schema
- ini hanya guardrail recall untuk display name

## 7. Final Reranked Slice

Setelah semua score dihitung:

```python
reranked = sorted(scored, key=lambda x: x["boosted"], reverse=True)
reranked = reranked[:MAX_CTX_DOCS]
```

Default:

```text
MAX_CTX_DOCS = 12
```

Jadi:

- sparse bisa menghasilkan sampai 20 hit
- dense bisa mengambil 12 atau 28 candidate
- tapi evidence utama yang dipakai setelah fusion hanya top 12 reranked docs

`state["ranked_cols"]` juga diisi dari `reranked` ini.

## 8. Filter Relevansi Kolom

Sesudah reranking, sistem belum langsung memasukkan semua kolom ke context.

Ia memfilter lagi dengan:

```python
_is_column_relevant_to_query(column, table, question, qtype, metadata)
```

Rule yang paling penting:

1. Kolom ID / identifying columns selalu boleh lewat.
2. Kolom yang disebut eksplisit lewat alias/variant metadata akan lolos.
3. Deskripsi kolom bisa membantu lolos kalau ada overlap kata yang spesifik.
4. Untuk query aggregate count-only, metric column yang tidak diminta bisa dibuang.
5. Untuk query aggregate non-count, metric column cenderung dipertahankan.
6. Untuk lookup, name/display/filter columns lebih mudah lolos.

Filter ini dipakai di 2 tempat:

- saat membangun `allowed`
- saat membangun `cards`

Jadi kolom bisa saja tinggi di rerank, tetapi tetap tidak tampil di context
akhir kalau dianggap tidak relevan dengan intent.

## 9. Scoring Tabel

Setelah `allowed` awal terbentuk, retrieval menghitung score tabel dengan:

```python
score_tables_from_reranked(reranked, token_for_score)
```

Signal utama yang dipakai:

- sparse hit pada candidate tabel itu
- apakah nama/synonym tabel disebut di question
- overlap token deskripsi
- table type dari schema analyzer
- apakah tabel bridge perlu dipenalti
- apakah query butuh numeric detail dari bridge

Konstanta default:

- `TABLE_SCORE_SPARSE_HIT = 0.35`
- `TABLE_SCORE_TABLE_MENTION = 1.3`
- `TABLE_SCORE_MENTION_DIM = 0.9`
- `TABLE_SCORE_MENTION_TXN = 0.7`
- `TABLE_SCORE_MENTION_OTHER = 0.8`
- `TABLE_SCORE_BRIDGE_PENALTY = -0.5`
- `TABLE_SCORE_BRIDGE_NUMERIC_BOOST = 0.3`

Artinya, reranked column digunakan sebagai evidence untuk memutuskan tabel mana
yang pantas masuk context final.

## 10. Single-Table Lock

Kalau query kelihatan single-table, retrieval bisa mengunci context ke satu
tabel utama.

Deteksi awal:

- qtype harus `lookup` atau `aggregate`
- ada primary table eksplisit
- `_has_join_intent()` harus false

Lalu confidence lock ditentukan oleh:

```python
_single_table_lock_is_high_confidence(...)
```

Ia memeriksa:

- gap score tabel utama vs tabel kedua
- jumlah sparse hits dari tabel lain
- tier evidence retrieval

Default penting:

- `RETRIEVAL_SINGLE_LOCK_MIN_SCORE_GAP = 0.6`
- `RETRIEVAL_SINGLE_LOCK_MAX_NON_PRIMARY_SPARSE_HITS = 4`
- `RETRIEVAL_SINGLE_LOCK_FALLBACK_TOPK = 1`

Kalau confidence rendah, sistem tidak memaksa hard lock. Ia fallback ke primary
table plus top table tambahan.

## 11. Multi-Table Selection dan Bridge Inference

Kalau query tidak dikunci ke satu tabel, retrieval memilih tabel final dari:

- top scored tables
- tabel yang punya kolom di reranked result
- safeguard temporal table untuk query waktu
- bridge table hasil inferensi FK path

Base limit:

```text
MAX_TOP_TABLES = 3
```

Ada juga `score_floor` yang bisa berubah, terutama untuk aggregate:

- count-only aggregate cenderung lebih ketat
- metric aggregate bisa lebih longgar

### Bridge inference

Jika ada beberapa tabel terpilih, retrieval dapat mencari connector path lewat
FK graph.

Mekanisme penting:

- FK graph dibangun dari schema metadata + extracted metadata hints
- path dicari dengan BFS
- bridge table di tengah path bisa ditambahkan ke `top_tables`
- FK path columns bisa ikut dimasukkan ke `allowed`

Tujuannya supaya generator SQL tetap punya jalur join yang valid.

## 12. ALLOWED IDENTIFIERS, CARDS, dan alias_map

### 12.1 `ALLOWED IDENTIFIERS`

Blok ini dibangun dari `filtered_allowed`.

Contoh:

```text
### ALLOWED IDENTIFIERS
- customers: customerNumber, customerName, country
- orders: orderNumber, customerNumber, orderDate
```

Ini adalah whitelist utama untuk generator SQL.

### 12.2 `CARDS`

`_build_cards()` mengambil kolom dari reranked result yang masih lolos filter.

Deskripsi kolom diambil dengan prioritas:

1. structured description dari metadata/schema
2. fallback humanized column name

Stats ditambahkan bila ada, misalnya:

- sample value
- range
- mean
- distinct count

Kalau ada kolom di `filtered_allowed` yang belum punya card, sistem dapat
menambahkan card sederhana lewat injection.

### 12.3 `_INTERNAL_HINT.alias_map`

Alias map dibangun dari alias metadata yang telah dinormalisasi.

Contoh:

```json
{"alias_map": {"customers": {"client name": "customerName"}}}
```

Ini membantu fase downstream saat harus memetakan istilah user ke identifier
schema yang valid.

## 13. Telemetry Retrieval

Retrieval juga menyimpan snapshot ringkas:

- jumlah sparse candidate
- jumlah dense candidate
- jumlah reranked candidate
- jumlah tabel/kolom sebelum dan sesudah filter
- preview sparse/dense/rerank
- evidence score dan evidence tier

Evidence strength dihitung oleh:

```python
_compute_retrieval_evidence_strength(snapshot)
```

Tier yang mungkin:

- `strong`
- `moderate`
- `weak`

Telemetry ini dipakai, antara lain, untuk membantu keputusan single-table lock.

## 14. Contoh End-to-End

Contoh question:

```text
Show customer names and their order dates
```

Kemungkinan alurnya:

1. `qtype` terbaca sebagai lookup.
2. Sparse query menjadi token seperti `customer name order date`.
3. BM25 menemukan evidence seperti:
   - `customers.customerName`
   - `customers.customerNumber`
   - `orders.orderDate`
   - `orders.customerNumber`
4. Dense probe menambahkan hint schema dan query ke Chroma.
5. Dense result memberi candidate semantic terdekat.
6. Reranking menaikkan candidate yang:
   - dekat secara semantic
   - juga kena sparse
   - sesuai intent name/date
7. Table scoring mempertahankan `customers` dan `orders`.
8. Jika perlu, FK path menjaga kolom join seperti `customerNumber`.
9. Context akhir bisa menjadi:

```text
### ALLOWED IDENTIFIERS
- customers: customerNumber, customerName
- orders: customerNumber, orderDate, orderNumber

### CARDS
- table: customers | column: customerName | desc: customer full name
- table: customers | column: customerNumber | desc: customer identifier
- table: orders | column: orderDate | desc: order date
- table: orders | column: customerNumber | desc: foreign key to customer

### _INTERNAL_HINT
{"alias_map": {"customers": {"customer name": "customerName"}}}
```

## 15. Hal yang Paling Penting untuk Dipahami

Retrieval Anda bukan sekadar:

```text
BM25 + embedding lalu ambil top-k
```

Secara aktual, retrieval Anda adalah:

```text
hybrid candidate generation
-> fusion berbasis score
-> intent-aware metadata boosting
-> column relevance filtering
-> table scoring
-> single-table / multi-table control
-> FK-aware context shaping
-> whitelist + cards + alias map
```

Dengan kata lain, reranking di sistem Anda bukan hanya "sort by embedding
distance". Ia adalah proses penggabungan beberapa sinyal:

- semantic similarity dari dense retrieval
- lexical confirmation dari sparse retrieval
- intent alignment dari metadata boost
- structural control lewat table scoring dan FK path logic

## 16. Nilai Default Penting

| Config | Default | Fungsi |
| --- | ---: | --- |
| `TOP_K` | 5 | knob umum top-k |
| `RETRIEVAL_BASE_K_FLOOR` | 12 | minimum dense candidates |
| `RETRIEVAL_DENSE_K_FLOOR` | 28 | dense candidates untuk lookup-name recall |
| `SPARSE_MAX_HITS` | 20 | maksimum sparse BM25 hits |
| `SPARSE_BONUS` | 0.75 | bonus jika dense doc juga kena sparse |
| `MAX_CTX_DOCS` | 12 | cap final reranked docs |
| `MAX_TOP_TABLES` | 3 | base jumlah tabel teratas |
| `RETRIEVAL_SINGLE_LOCK_MIN_SCORE_GAP` | 0.6 | gap minimum untuk hard single-table lock |
| `RETRIEVAL_SINGLE_LOCK_MAX_NON_PRIMARY_SPARSE_HITS` | 4 | batas sparse hit non-primary |
| `RETRIEVAL_SINGLE_LOCK_FALLBACK_TOPK` | 1 | fallback top-k saat lock lemah |
| `RETRIEVAL_AUTO_INJECT_BONUS_DELTA` | 0.05 | bonus kecil untuk preferred name auto-inject |
| `TABLE_SCORE_TABLE_MENTION` | 1.3 | boost jika tabel disebut |
| `TABLE_SCORE_BRIDGE_PENALTY` | -0.5 | penalti bridge table default |

## 17. Satu Kalimat Paling Singkat

Proses retrieval Anda mengambil candidate schema dari BM25 dan Chroma, lalu
mererank-nya dengan gabungan distance, sparse evidence, dan metadata-driven
intent boost, kemudian menyaring dan membentuk context final berupa whitelist
tabel/kolom yang aman untuk generator SQL.
