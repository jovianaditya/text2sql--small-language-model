import os
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import (
    TOP_K,
    SPARSE_BONUS,
    NAME_QUERY_BONUS,
    ID_QUERY_BONUS,
    MAX_CTX_DOCS,
    SPARSE_MAX_HITS,
    SPARSE_DEBUG,
    RETRIEVE_DEBUG,
)

from .state import RAGState
from .chroma_embedding import build_vector_store
from .utils import _print_banner, _meta_table, _meta_column, read_jsonl, _parse_str_or_list_lower, _extract_roles_aliases, _extract_aliases_for_map
from .helpers import _is_productline_lookup, _is_simple_orders_lookup, _wants_customer_name_only, _wants_product_name_only, _get_qtype, build_dynamic_probe

# ======================================================================
# ========================== RETRIEVAL POLICY ==========================
# ======================================================================
COLUMN_PRIORITY_CONFIG = [
    {
        "id": "customers.primary_name",
        "tables": ["customers"],
        "role_any": ["name"],
        "alias_any": [
            "customer name",
            "nama customer",
            "nama pelanggan",
            "company name",
            "nama perusahaan",
            "account name",
        ],
        "column_contains": ["customername"],
        "query_terms_any": ["name", "nama"],
        "query_terms_also_any": ["customer", "customers", "pelanggan"],
        "boost": 0.4,
    },
]

# ======================================================================
# ========================== CONTEXT BUILDER ===========================
# ======================================================================
def _build_allowed_block(filtered_allowed: Dict[str, List[str]]) -> str:
    """
    Build blok ### ALLOWED IDENTIFIERS dari dict table -> list kolom.
    """
    lines = []
    for t, cols in filtered_allowed.items():
        uniq = sorted(set(cols))
        lines.append(f"- {t}: {', '.join(uniq)}")
    body = "\n".join(lines) if lines else "- (none)"
    return "### ALLOWED IDENTIFIERS\n" + body

def _build_cards(
    filtered_allowed: Dict[str, List[str]],
    reranked: List[Dict[str, Any]],
    max_cards: int = 15,
    max_inject: int = 12,
    desc_max_len: int = 100,
) -> str:
    """
    Build kartu-kartu preview table/column untuk context LLM.

    Optimasi (tanpa mengubah logic):
    - Batasi jumlah kartu dari reranked (max_cards)
    - Pendekkan description (desc_max_len)
    - Batasi jumlah "inject" kolom yang belum muncul (max_inject)
      supaya context tidak membengkak untuk kasus easy/medium.

    Semantik tetap ada (desc tetap ada), tapi token input turun signifikan.
    """
    cards: List[str] = []
    present = set()

    # ------------------------------------------------------------
    # 1) Kartu dari reranked (dibatasi)
    # ------------------------------------------------------------
    for it in (reranked or [])[:max_cards]:
        md = it.get("meta") or {}
        table = _meta_table(md)
        column = _meta_column(md)
        if not table or not column:
            continue
        if table not in filtered_allowed:
            continue

        desc = (md.get("description") or "").strip()
        if len(desc) > desc_max_len:
            desc = desc[:desc_max_len] + "..."

        cards.append(f"- table: {table} | column: {column} | desc: {desc}")
        present.add((table.lower(), column.lower()))

    # ------------------------------------------------------------
    # 2) Inject kolom penting yang belum muncul (dibatasi)
    # ------------------------------------------------------------
    injected = 0
    for tbl, cols in (filtered_allowed or {}).items():
        if injected >= max_inject:
            break

        for col in cols:
            if injected >= max_inject:
                break

            key = (tbl.lower(), col.lower())
            if key in present:
                continue

            short_desc = ""
            if tbl.lower() == "customers" and col.lower() == "customername":
                short_desc = "customer display name / company name"

            cards.append(f"- table: {tbl} | column: {col} | desc: {short_desc}")
            present.add(key)
            injected += 1

    return "\n".join(cards)

# ======================================================================
# =========================== SPARSE (BM25) ============================
# ======================================================================
_BM25_CACHE: dict[str, Any] = {}
# Base stopwords dari scikit-learn (bahasa Inggris)
_BASE_STOPWORDS = set(ENGLISH_STOP_WORDS)

for w in ["name", "names"]:
    _BASE_STOPWORDS.discard(w)

# Tambahan stopwords domain + Bahasa Indonesia
_EXTRA_STOPWORDS = {
    # kata2 query umum / “filler”
    "show", "all", "list", "their", "that", "have", "no",
    "include", "includes", "including", "sort", "sorted", "sorting",
    "result", "results", "record", "records", "row", "rows",
    "display", "return", "get", "give", "find", "fetch", "please", "kindly",
    "alphabetical", "alphabetically", "ascending", "descending", "distinct",

    # Bahasa Indonesia dasar
    "yang", "dan", "atau", "ke", "dari", "di", "itu", "ini",
    "tampilkan", "daftar",
}

# FINAL stopwords: English (sklearn) + ekstra custom
_STOPWORDS = _BASE_STOPWORDS | _EXTRA_STOPWORDS

def _normalize_text(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"(\d)[,\.](\d{3})(\b|$)", r"\1\2", txt)
    txt = re.sub(r"(\w)'s\b", r"\1", txt)
    txt = txt.replace("'", "")
    txt = re.sub(r"[^a-z0-9_]+", " ", txt)
    return txt.strip()

def _tokenize_bm25_query(q: str) -> List[str]:
    """
    Normalisasi + tokenisasi untuk BM25:
    - lowercase + bersihkan non-alfanumerik
    - buang stopword
    - stem plural sederhana:
      * countries -> country
      * customers -> customer
      (hanya kalau panjang > 3 supaya 'as' / 'is' nggak ke-cut)
    """
    s = _normalize_text(q)
    raw_tokens = s.split()
    toks: List[str] = []

    for t in raw_tokens:
        if not t:
            continue

        # --- plural stemming ringan ---
        if len(t) > 4 and t.endswith("ies"):
            # countries -> country
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            # customers -> customer, orders -> order
            t = t[:-1]

        if t and t not in _STOPWORDS:
            toks.append(t)

    return toks

def build_sparse_query_from_question(q: str) -> str:
    """
    Build query string untuk BM25 dari pertanyaan user.
    Di sini bisa tambahkan hint manual (synonym → canonical).
    """
    toks = _tokenize_bm25_query(q)
    q_l = (q or "").lower()

    # Hint: "list price" biasanya maksudnya MSRP
    if "list price" in q_l and "msrp" not in toks:
        toks.append("msrp")

    return " ".join(toks)

def _doc_tokens_from_meta(meta: Dict[str, Any]) -> List[str]:
    table = (_meta_table(meta) or "").lower()
    column = (_meta_column(meta) or "").lower()
    desc = str(meta.get("description") or "").lower()

    # aliases/roles dari DataHub → parse dengan benar (string JSON / list)
    aliases = _parse_str_or_list_lower(meta.get("tags_alias") or meta.get("aliases") or [])
    roles   = _parse_str_or_list_lower(meta.get("tags_role")  or meta.get("roles")   or [])

    extra: List[str] = []
    has_semantic = bool(aliases or roles)

    # ---------- TABLE-LEVEL SYNONYMS ----------
    # productlines spesial: selalu dibantu, karena query natural user sering "product lines / kategori"
    if table == "productlines":
        extra += ["product", "products", "product line", "product lines", "line", "category", "kategori"]
    else:
        # tabel lain: cuma kalau DataHub nggak supply alias/role sama sekali
        if not has_semantic:
            if table == "products":
                extra += ["product", "products", "item", "items", "barang", "produk"]
            elif table == "customers":
                extra += ["customer", "customers", "pelanggan", "client", "account"]
            elif table == "orders":
                extra += ["order", "orders", "pesanan", "transaksi"]
            elif table == "payments":
                extra += ["payment", "payments", "pembayaran"]
            # boleh tambah lain kalau perlu, tapi keep it small

    # ---------- COLUMN-LEVEL SYNONYMS (bridging kuat) ----------
    if column == "msrp":
        extra += ["price", "list price", "retail price", "harga jual"]
    elif column == "priceeach":
        extra += ["unit price", "selling price", "harga satuan"]
    elif column == "quantityinstock":
        extra += ["stock", "stok", "qty", "quantity", "on hand"]
    elif column == "quantityordered":
        extra += ["qty", "quantity ordered", "jumlah pesanan"]
    elif column == "paymentdate":
        extra += ["payment date", "tanggal pembayaran"]
    elif column == "amount":
        extra += ["amount", "nominal", "jumlah pembayaran"]

    # helper khusus description
    if column == "htmldescription":
        extra += ["html", "description", "html description"]
    if column == "textdescription":
        extra += ["text", "description", "text description"]

    bag = " ".join([table, column, desc] + aliases + roles + extra)
    return _tokenize_bm25_query(bag)

def jsonl_sparse_search(q: str, jsonl_paths: List[str], max_hits: int = 20) -> List[str]:
    """
    Kembalikan daftar candidate IDs (field_urn/urn fallback f"{source}:{table}::{column}") berdasar BM25.
    Di-cache per (paths hash) agar cepat.
    """
    # cache key by files + mtime
    sig = []
    for p in jsonl_paths:
        try:
            st = os.stat(p)
            sig.append(f"{p}:{int(st.st_mtime)}:{int(st.st_size)}")
        except Exception:
            sig.append(p)
    cache_key = "||".join(sorted(sig))

    if cache_key not in _BM25_CACHE:
        tokenized_corpus: List[List[str]] = []
        ids: List[str] = []

        for path in jsonl_paths:
            p = Path(path)
            if not p.exists():
                continue
            srcname = p.name

            for line_no, (raw, obj) in enumerate(read_jsonl(path), 1):
                # ---- CASE 1: line bukan dict → coba list-of-dict ----
                if not isinstance(obj, dict):
                    try:
                        maybe = json.loads(raw)
                    except Exception:
                        maybe = None

                    if isinstance(maybe, list):
                        for i, item in enumerate(maybe):
                            if not isinstance(item, dict):
                                continue
                            table  = (_meta_table(item) or "").lower()
                            column = (_meta_column(item) or "").lower()

                            doc_id_raw = (
                                item.get("field_urn")
                                or item.get("urn")
                                or f"{srcname}:{table}::{column}"
                            )
                            doc_id = str(doc_id_raw)
                            tokens = _doc_tokens_from_meta(item)
                            tokenized_corpus.append(tokens)
                            ids.append(doc_id)
                        continue  # lanjut line berikutnya
                    # kalau bukan dict & bukan list-of-dict → skip line
                    continue

                # ---- CASE 2: obj sudah dict normal ----
                table  = (_meta_table(obj) or "").lower()
                column = (_meta_column(obj) or "").lower()

                doc_id_raw = (
                    obj.get("field_urn")
                    or obj.get("urn")
                    or f"{srcname}:{table}::{column}"
                )
                doc_id = str(doc_id_raw)

                tokens = _doc_tokens_from_meta(obj)
                tokenized_corpus.append(tokens)
                ids.append(doc_id)

        if not tokenized_corpus:
            _BM25_CACHE[cache_key] = {"bm25": None, "ids": []}
        else:
            _BM25_CACHE[cache_key] = {
                "bm25": BM25Okapi(tokenized_corpus),
                "ids": ids,
            }

    entry = _BM25_CACHE[cache_key]
    if not entry["bm25"]:
        return []

    query_tokens = _tokenize_bm25_query(q)
    scores = entry["bm25"].get_scores(query_tokens)

    # <<< FIX: paksa jadi list supaya 'not scores' nggak pakai numpy boolean >>>
    if not isinstance(scores, list):
        scores = list(scores)

    # NEW: kalau semua skor == 0 → anggap sparse "nggak tau apa-apa"
    if not scores or max(scores) <= 0:
        if SPARSE_DEBUG:
            print("[SPARSE] all scores are 0 → returning no sparse hits")
        return []

    ranked = sorted(zip(entry["ids"], scores), key=lambda x: x[1], reverse=True)
    out = [doc_id for doc_id, _ in ranked[:max_hits]]

    if SPARSE_DEBUG:
        print("[SPARSE] top hits:")
        for doc_id, sc in ranked[:min(max_hits, 10)]:
            print(f"  - {doc_id}  score={sc:.3f}")

    return out

def _doc_matches_sparse(md: Dict[str, Any], sparse_ids: List[str]) -> bool:
    if not md or not sparse_ids:
        return False

    sid_lower = {x.lower() for x in sparse_ids}

    urn = str(md.get("field_urn") or md.get("urn") or "").lower()
    if urn and urn in sid_lower:
        return True

    table  = (_meta_table(md) or "").lower()
    column = (_meta_column(md) or "").lower()
    if table and column:
        suffix = f":{table}::{column}"
        return any(x.lower().endswith(suffix) for x in sparse_ids)

    return False

 # =========================
# ======= SCORING =========
# =========================
def apply_column_priority_boost(
    q_l: str,
    table: str,
    col_name: str,
    roles: List[str],
    aliases: List[str],
) -> float:
    bonus = 0.0
    if any(k in q_l for k in ["name", "nama"]):
        if any(r == "name" or "name" in r for r in roles) or "name" in col_name:
            bonus += NAME_QUERY_BONUS
    if table.lower() == "customers":
        mentions_customer = any(k in q_l for k in ["customer", "customers", "pelanggan"])
        asks_id_like = any(k in q_l for k in [" id", "id ", " id,", "identifier", "kode", "code", "customer number"])
        if mentions_customer and not asks_id_like:
            if "customername" in col_name or "customer_name" in col_name:
                bonus += 0.6
            if "customernumber" in col_name or col_name.endswith("number"):
                bonus -= 0.35
    if table.lower() == "products":
        mentions_product = any(k in q_l for k in ["product", "products", "produk"])
        asks_id_like = any(k in q_l for k in [" id", "id ", " id,", "identifier", "kode", "code", "product code"])
        # Kalau user ngomong produk tapi nggak eksplisit minta ID/kode,
        # dorong productName, "turunin" productCode sedikit.
        if mentions_product and not asks_id_like:
            if "productname" in col_name or "product_name" in col_name:
                bonus += 0.5   # boleh kamu atur naik/turun
            if "productcode" in col_name or col_name.endswith("code"):
                bonus -= 0.3   # penalti ringan supaya kalah dari nama
    if any(k in q_l for k in [" id ", "id ", " id,", "identifier", "kode", "code"]):
        if (
            any(r == "id" or "id" in r for r in roles)
            or "number" in col_name
            or col_name.lower().endswith("id")
        ):
            bonus += ID_QUERY_BONUS

    col_lower = col_name.lower()
    role_set = set(roles)
    alias_set = set(aliases)
    for rule in COLUMN_PRIORITY_CONFIG:
        tables = [t.lower() for t in rule.get("tables", []) or []]
        if tables and table.lower() not in tables:
            continue

        role_any = rule.get("role_any") or []
        if role_any and not any(r in role_set for r in role_any):
            continue

        alias_any = rule.get("alias_any") or []
        if alias_any and not any(a in alias_set for a in alias_any):
            continue

        column_contains = rule.get("column_contains") or []
        if column_contains and not any(sub in col_lower for sub in column_contains):
            continue

        qt_any = rule.get("query_terms_any") or []
        if qt_any and not any(t in q_l for t in qt_any):
            continue

        qt_also = rule.get("query_terms_also_any") or []
        if qt_also and not any(t in q_l for t in qt_also):
            continue

        bonus += float(rule.get("boost", 0.0))

    return bonus

def score_tables_from_reranked(
    reranked: List[Dict[str, Any]],
    query_tokens: List[str],
) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    qt = [t.lower() for t in query_tokens]
    q_l = " ".join(qt)

    # entity hints dari token
    mentions_customer = ("customer" in q_l) or ("customers" in q_l) or ("pelanggan" in q_l)
    mentions_product  = ("product" in q_l) or ("products" in q_l) or ("produk" in q_l)
    mentions_order    = "order" in q_l or "orders" in q_l
    mentions_payment  = ("payment" in q_l) or ("payments" in q_l) or ("pembayaran" in q_l)

    # tambahan sesuai tabel classicmodels:
    mentions_office = any(
        k in q_l
        for k in ["office", "offices", "kantor", "territory", "territories", "region", "regional"]
    )
    mentions_productline = any(
        k in q_l
        for k in ["product line", "product lines", "kategori", "category", "lini produk", "line produk"]
    )
    mentions_orderdetails = any(
        k in q_l
        for k in ["order detail", "order details", "line item", "line items", "item pesanan", "detail pesanan"]
    )
    mentions_employee = any(
        k in q_l
        for k in ["employee", "employees", "karyawan", "sales rep", "salesrep"]
    )
    penalize_orderdetails_default = mentions_order and not mentions_orderdetails
    
    for it in reranked:
        md = it.get("meta") or {}
        tbl = _meta_table(md)
        if not tbl:
            continue
        desc = (md.get("description") or "").lower()
        base = 0.0

        tl = tbl.lower()

        # sinyal sparse
        if it.get("sparse"):
            base += 0.5

        # tabel disebut eksplisit di query
        if tl in q_l:
            base += 1.0

        # token ada di description
        for t in qt:
            if t and t in desc:
                base += 0.3

        # entity-specific boosts utama
        if mentions_customer and tl == "customers":
            base += 0.9
        if mentions_product and tl == "products":
            base += 0.9
        if mentions_order and tl == "orders":
            base += 0.7
        if mentions_payment and tl == "payments":
            base += 0.7

        # entity-specific boosts tambahan
        if mentions_office and tl == "offices":
            base += 0.8
        if mentions_productline and tl == "productlines":
            base += 0.8
        if mentions_orderdetails and tl == "orderdetails":
            base += 0.8
        if penalize_orderdetails_default and tl == "orderdetails":
            base -= 0.4   # angka ini bisa kamu tuning
        if mentions_employee and tl == "employees":
            base += 0.8

        scores[tbl] += base

    return scores

# ======================================================================
# =========================== RETRIEVE NODE ============================
# ======================================================================
def _has_join_intent(question: str) -> bool:
    """
    True jika pertanyaan terlihat butuh join / multi-entity.
    Ini bukan mengubah logic agent, hanya memilih konteks yang lebih kecil.
    """
    q = (question or "").lower()

    join_signals = [
        "join", "gabung", "combine", "combine with", "along with", "together with",
        "and their", "with their", "beserta", "bersama", "dengan",
        "for each", "per ", "setiap",
    ]
    # multi-entity hints (classicmodels)
    multi_entity = [
        ("customer", "order"),
        ("customer", "payment"),
        ("order", "product"),
        ("order", "order detail"),
        ("order", "orderdetails"),
        ("employee", "office"),
        ("employee", "customer"),
        ("office", "employee"),
    ]

    if any(s in q for s in join_signals):
        return True

    for a, b in multi_entity:
        if a in q and b in q:
            return True

    return False

def _detect_primary_entity_table(question: str) -> Optional[str]:
    """
    Tentukan 1 tabel utama bila pertanyaan menyebut entitas dengan jelas.
    Prioritas: frasa eksplisit yang kuat.
    """
    q = (question or "").lower()

    # urutkan dari yang paling spesifik/kuat
    if "product line" in q or "product lines" in q or "lini produk" in q or "line produk" in q:
        return "productlines"
    if "order detail" in q or "order details" in q or "line item" in q or "line items" in q or "detail pesanan" in q:
        return "orderdetails"
    if "payment" in q or "payments" in q or "pembayaran" in q:
        return "payments"
    if "office" in q or "offices" in q or "kantor" in q or "territory" in q:
        return "offices"
    if "employee" in q or "employees" in q or "karyawan" in q:
        return "employees"
    if "order" in q or "orders" in q or "pesanan" in q or "transaksi" in q:
        return "orders"
    if "customer" in q or "customers" in q or "pelanggan" in q:
        return "customers"
    if "product" in q or "products" in q or "produk" in q:
        return "products"

    return None

def _should_lock_single_table_ctx(question: str, qtype: str) -> bool:
    """
    Kapan kita boleh mengunci konteks ke 1 tabel?
    - qtype lookup/aggregate/rank masih boleh single-table (rank sering single-table juga)
    - TIDAK ada join intent
    - ada entitas utama yang jelas
    """
    if qtype not in {"lookup", "aggregate", "rank"}:
        return False
    if _has_join_intent(question):
        return False
    return _detect_primary_entity_table(question) is not None

def retrieve_node(state: RAGState) -> RAGState:
    # -----------------------------
    # 0) Build vector store
    # -----------------------------
    col = build_vector_store()

    q = (state["question"] or "").strip()
    q_l = q.lower()
    qtype = _get_qtype(state)

    wants_customer_no_id = _wants_customer_name_only(q)
    productline_lookup = _is_productline_lookup(q)
    orders_simple_lookup = _is_simple_orders_lookup(q)

    jsonl_paths = state.get("jsonl_paths", [])

    # -----------------------------
    # 1) Sparse query build + BM25 search
    # -----------------------------

    sparse_query_clean = build_sparse_query_from_question(q)
    sparse_tokens = _tokenize_bm25_query(sparse_query_clean)


    sparse_field_urns = jsonl_sparse_search(
        sparse_query_clean,
        jsonl_paths,
        max_hits=SPARSE_MAX_HITS
    )

    # --- get sparse metas sekali, lalu pakai untuk pretty + flags ---
    sparse_docs_pretty: List[str] = []
    sparse_metas: List[Dict[str, Any]] = []

    if sparse_field_urns:
        try:
            got = col.get(ids=sparse_field_urns, include=["metadatas"])
            ids = got.get("ids") or []
            metas = got.get("metadatas") or []

            # Chroma kadang balikin list-of-list; normalisasi jadi flat list
            if metas and isinstance(metas[0], list):
                metas = metas[0]

            meta_by_id = {i: (m or {}) for i, m in zip(ids, metas)}

            for doc_id in sparse_field_urns:
                md = meta_by_id.get(doc_id, {}) or {}
                sparse_metas.append(md)

                tbl = _meta_table(md) or "?"
                c = _meta_column(md) or "?"
                sparse_docs_pretty.append(f"table={tbl}  column={c}")

        except Exception as e:
            sparse_docs_pretty = list(sparse_field_urns or [])
            sparse_metas = []
            if RETRIEVE_DEBUG:
                print(f"[WARN] Sparse meta fetch failed: {e!r}")

    # flag customerName dari metadata (lebih stabil daripada parsing URN)
    sparse_has_customername = any(
        ((_meta_table(m) or "").lower() == "customers")
        and ((_meta_column(m) or "").lower() == "customername")
        for m in sparse_metas
        if isinstance(m, dict)
    )

    base_k = max(12, TOP_K)
    dense_k = base_k
    if wants_customer_no_id and not sparse_has_customername:
        dense_k = max(28, base_k)

    # -----------------------------
    # 2) Dense query
    # -----------------------------
    query_text = build_dynamic_probe(q, qtype=qtype)

    res = col.query(
        query_texts=[query_text],
        n_results=dense_k,
        include=["documents", "metadatas", "distances"],
    )

    # -----------------------------
    # 3) Unpack dense results
    # -----------------------------
    dense_docs = (res.get("documents") or [[]])[0]
    if dense_docs and isinstance(dense_docs[0], list):
        dense_docs = dense_docs[0]

    dense_metas = (res.get("metadatas") or [[]])[0]
    if dense_metas and isinstance(dense_metas[0], list):
        dense_metas = dense_metas[0]

    dense_dists = (res.get("distances") or [[]])[0]
    if dense_dists and isinstance(dense_dists[0], list):
        dense_dists = dense_dists[0]

    dense_ids = (res.get("ids") or [[]])[0]
    if dense_ids and isinstance(dense_ids[0], list):
        dense_ids = dense_ids[0]

    n = min(
        len(dense_docs),
        len(dense_metas),
        len(dense_dists) if dense_dists else len(dense_docs),
        len(dense_ids) if dense_ids else len(dense_docs),
    )
    dense_docs = dense_docs[:n]
    dense_metas = dense_metas[:n]
    dense_dists = (dense_dists or [0.0] * n)[:n]
    dense_ids = (dense_ids or [""] * n)[:n]

    dense_has_customername = any(
        ((_meta_table(md) or "").lower() == "customers")
        and ((_meta_column(md) or "").lower() in ("customername", "customer_name"))
        for md in dense_metas
    )

    # -----------------------------
    # 4) Scoring + rerank (dense + sparse bonus + priority boost)
    # -----------------------------
    scored = []
    for rank, (doc_text, md, dist, _id) in enumerate(zip(dense_docs, dense_metas, dense_dists, dense_ids)):
        base = -float(dist) if dist is not None else 0.0
        sparse_hit = _doc_matches_sparse(md, sparse_field_urns)

        table_name = _meta_table(md) or ""
        col_raw = _meta_column(md) or ""
        col_name = col_raw.lower()

        # parse roles & aliases (DRY helper)
        roles, aliases = _extract_roles_aliases(md)

        boosted = base + (SPARSE_BONUS if sparse_hit else 0.0)
        boosted += apply_column_priority_boost(
            q_l=q_l,
            table=table_name,
            col_name=col_name,
            roles=roles,
            aliases=aliases,
        )

        scored.append({
            "rank": rank,
            "id": _id,
            "doc": doc_text,
            "meta": md or {},
            "dist": float(dist) if dist is not None else 0.0,
            "base": base,
            "sparse": sparse_hit,
            "boosted": boosted,
        })

    # -----------------------------
    # 5) Auto-inject sangat terbatas (customerName)
    # -----------------------------
    if wants_customer_no_id and sparse_has_customername and not dense_has_customername:
        sparse_customername_urn = None
        for u in sparse_field_urns:
            if u.lower().endswith("::customername"):
                sparse_customername_urn = u
                break

        if sparse_customername_urn:
            present_ids = {item["id"] for item in scored if item.get("id")}
            if sparse_customername_urn not in present_ids:
                got = col.get(ids=[sparse_customername_urn])
                if got and got.get("ids"):
                    md = (got.get("metadatas") or [{}])[0]
                    doc = (got.get("documents") or [""])[0]

                    roles, aliases = _extract_roles_aliases(md)

                    table_name = _meta_table(md) or ""
                    col_raw = _meta_column(md) or ""
                    col_name = col_raw.lower()

                    max_boost = max((x["boosted"] for x in scored), default=0.0)
                    boosted = max_boost + 0.05
                    boosted += apply_column_priority_boost(
                        q_l=q_l,
                        table=table_name,
                        col_name=col_name,
                        roles=roles,
                        aliases=aliases,
                    )

                    scored.append({
                        "rank": len(scored),
                        "id": sparse_customername_urn,
                        "doc": doc,
                        "meta": md or {},
                        "dist": 0.0,
                        "base": 0.0,
                        "sparse": True,
                        "boosted": boosted,
                    })

    # -----------------------------
    # 6) Final reranked slice
    # -----------------------------
    reranked = sorted(scored, key=lambda x: x["boosted"], reverse=True)
    reranked = reranked[:MAX_CTX_DOCS]

    # -----------------------------
    # 6b) Final reranked slice
    # -----------------------------
    state["ranked_cols"] = [
    ((_meta_table(it.get("meta") or {}) or "").strip().lower(),
     (_meta_column(it.get("meta") or {}) or "").strip())
    for it in (reranked or [])
    if (_meta_table(it.get("meta") or {}) and _meta_column(it.get("meta") or {}))
    ]
    
    # -----------------------------
    # 7) Debug prints (existing behavior)
    # -----------------------------
    if RETRIEVE_DEBUG:
        line = _print_banner("RETRIEVE DEBUG")

        print(f"QUESTION      : {q}")
        print(f"DENSE QUERY   : {query_text}")
        print(f"SPARSE QUERY  : {sparse_query_clean}")
        print(f"TOKENS        : {sparse_tokens}")
        print(f"QTYPE         : {qtype}\n")

        # =========================
        # SPARSE CANDIDATES
        # =========================
        total_sparse = len(sparse_docs_pretty)
        max_show = min(7, total_sparse)
        print(f"SPARSE CANDIDATES (top {max_show} of {total_sparse}):")
        if total_sparse == 0:
            print("  <none>\n")
        else:
            print(f"{'':7}{'table':<14} {'column':<30}")
            for i, s in enumerate(sparse_docs_pretty[:max_show], start=1):
                try:
                    _, rest = s.split("table=", 1)
                    table_part, col_part = rest.split("column=", 2)
                    table = table_part.strip() or "?"
                    column = col_part.strip() or "?"
                except Exception:
                    table = "?"
                    column = s
                print(f"  [{i:02d}] {table:<14} {column:<30}")
            if total_sparse > max_show:
                remaining = total_sparse - max_show
                print(f"  ... (+{remaining} more)")
            print()

        # =========================
        # DENSE CANDIDATES
        # =========================
        total_dense = len(dense_metas)
        max_show = min(7, total_dense)
        print(f"DENSE CANDIDATES (top {max_show} of {total_dense}):")
        if total_dense == 0:
            print("  <none>\n")
        else:
            print(f"{'':7}{'table':<14} {'column':<30}")
            for i in range(max_show):
                md = dense_metas[i] or {}
                tbl = _meta_table(md) or "?"
                col_ = _meta_column(md) or "?"
                print(f"  [{i+1:02d}] {tbl:<14} {col_:<30}")
            if total_dense > max_show:
                remaining = total_dense - max_show
                print(f"  ... (+{remaining} more)")
            print()

        # =========================
        # RERANKED RESULT
        # =========================
        total_rerank = len(reranked)
        max_show = min(7, total_rerank)
        print(f"RERANKED RESULT (top {max_show} of {total_rerank}):")
        if total_rerank == 0:
            print("  <no reranked candidates>")
        else:
            print(f"{'':7}{'table':<14} {'column':<30} {'sparse':>8}")
            for i, item in enumerate(reranked[:max_show], start=1):
                md = item.get("meta") or {}
                tbl = _meta_table(md) or "?"
                col_ = _meta_column(md) or "?"
                sparse_flag = "✓" if item.get("sparse") else "✗"
                print(f"  [{i:02d}] {tbl:<14} {col_:<30} {sparse_flag:>8}")
            if total_rerank > max_show:
                remaining = total_rerank - max_show
                print(f"  ... (+{remaining} more)")
        print(f"{line}\n")

    # -----------------------------
    # 8) Build allowed + alias_map (existing)
    # -----------------------------
    allowed = defaultdict(list)
    alias_map = defaultdict(dict)

    for item in reranked:
        md = item["meta"] or {}
        t = _meta_table(md)
        c = _meta_column(md)
        if not (t and c):
            continue

        t_key = t.strip().lower()   # canonical
        c_key = c.strip()           # keep original case

        allowed[t_key].append(c_key)

        aliases2 = _extract_aliases_for_map(md)
        for al in aliases2:
            al = al.strip()
            if al:
                alias_map[t_key][al] = c_key

    # -----------------------------
    # 9) Table scoring + filtered_allowed (existing)
    # -----------------------------
    token_for_score = _tokenize_bm25_query(sparse_query_clean)
    table_scores = score_tables_from_reranked(reranked, token_for_score)

    # Rule existing (tetap dipertahankan)
    if qtype == "lookup" and wants_customer_no_id:
        filtered_allowed = {t: cols for t, cols in allowed.items() if t.lower() == "customers"}

    elif qtype == "lookup" and productline_lookup:
        only_pl = {t: cols for t, cols in allowed.items() if t.lower() == "productlines"}
        filtered_allowed = only_pl if only_pl else dict(allowed)

    elif qtype == "lookup" and orders_simple_lookup:
        only_orders = {t: cols for t, cols in allowed.items() if t.lower() == "orders"}
        filtered_allowed = only_orders if only_orders else dict(allowed)

    else:
        if _should_lock_single_table_ctx(question=q, qtype=qtype):
            primary = _detect_primary_entity_table(q)  # lower-case
            only_primary = {t: cols for t, cols in allowed.items() if t.lower() == (primary or "")}

            if only_primary:
                filtered_allowed = only_primary
            else:
                if table_scores:
                    sorted_tbls = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
                    top1 = sorted_tbls[0][0]
                    filtered_allowed = {t: cols for t, cols in allowed.items() if t == top1}
                else:
                    filtered_allowed = dict(allowed)

        else:
            if table_scores:
                sorted_tbls = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
                top_tables = {name for name, score in sorted_tbls[:2] if score > 0.0}
                if top_tables:
                    filtered_allowed = {t: cols for t, cols in allowed.items() if t in top_tables}
                else:
                    filtered_allowed = dict(allowed)
            else:
                filtered_allowed = dict(allowed)

    if wants_customer_no_id:
        for t_key in list(filtered_allowed.keys()):
            if t_key.lower() == "customers":
                if "customerName" not in filtered_allowed[t_key]:
                    filtered_allowed[t_key].append("customerName")

    # -----------------------------
    # 10) Context build (existing)
    # -----------------------------
    allowed_block = _build_allowed_block(filtered_allowed)
    cards_text = _build_cards(
        filtered_allowed,
        reranked,
        max_cards=15,
        max_inject=12,
        desc_max_len=100,
    )

    context = (allowed_block + "\n\n### CARDS\n" + cards_text).strip()
    context += f"\n\n### _INTERNAL_HINT\n{json.dumps({'alias_map': alias_map}, ensure_ascii=False)}"

    state["context"] = context
    return state