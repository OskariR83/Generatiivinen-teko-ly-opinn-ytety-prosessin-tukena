"""
RAG Pipeline: Docling + DPR + FAISS (CPU) + Viking-7B
Project: GENERATIIVINEN TEKOÄLY OPINNÄYTETYÖPROSESSIN TUKENA
Description:
    End-to-end RAG pipeline that:
    - uses Docling to preprocess and extract text from documents
    - embeds passages with DPR
    - stores vectors in FAISS (CPU)
    - generates an answer with Viking-7B using the retrieved context
"""

import os
import json
import numpy as np
import faiss
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)
from docling.document_converter import DocumentConverter

from transformers.utils import logging
logging.set_verbosity_error()

# ========================
# Step 1: Docling Preprocessing
# ========================
def process_with_doclin(file_path: str):
    """
    Run Docling on a single file (DOCX/PDF/TXT).
    - Converts the document to markdown-like text
    - Caches the cleaned version under docs/processed/
    - Splits the text into ~300-word chunks for embedding
    """
    raw_path = Path(file_path)
    processed_dir = Path("docs/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_file = processed_dir / f"{raw_path.stem}_clean.json"

    # Reuse existing processed file if it exists
    if output_file.exists():
        print(f"📂 Using cached Docling output: {output_file}")
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print(f"🧠 Processing document with Docling: {file_path}")
        converter = DocumentConverter()
        result = converter.convert(file_path)
        text_output = result.document.export_to_markdown()

        data = {"text": text_output}

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Cleaned document saved to: {output_file}")

    # Split text into chunks for DPR
    text_blocks = []
    if "sections" in data:
        # In case Docling returns structured sections
        for section in data["sections"]:
            text = section.get("text", "").strip()
            if text:
                text_blocks.append(text)
    elif "text" in data:
        text = data["text"]
        words = text.split()
        chunk_size = 300
        text_blocks = [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

    print(f"✅ Extracted {len(text_blocks)} text blocks from Docling output.")
    return text_blocks


# ========================
# Step 2: Build FAISS Index (CPU)
# ========================
def build_faiss_index_multi(base_docs_dir="docs/originals", index_path="docs/indexes/combined_index.faiss"):
    """
    Processes all documents under docs/originals/ with Docling,
    builds (or loads) a single FAISS index containing all text chunks,
    and saves a metadata mapping (which file each passage came from).

    Features:
    - Reuses existing processed files (docs/processed/*.json)
    - Caches the FAISS index (docs/indexes/*.faiss)
    - Automatically skips rebuild if originals haven't changed
    - Prints which document triggered a rebuild
    """

    from transformers import AutoTokenizer

    docs_path = Path(base_docs_dir)
    index_dir = Path("docs/indexes")
    index_dir.mkdir(parents=True, exist_ok=True)

    index_file = Path(index_path)
    meta_file = index_file.with_suffix(".meta.json")

    # =====================================================
    # ✅ CACHE CHECK: reuse index if originals haven't changed
    # =====================================================
    if index_file.exists() and meta_file.exists():
        originals = list(docs_path.glob("*"))
        index_mtime = index_file.stat().st_mtime

        # find any newer files
        changed = [f.name for f in originals if f.stat().st_mtime > index_mtime]
        if not changed:
            print(f"📂 Using existing FAISS index: {index_path}")
            index = faiss.read_index(str(index_file))

            # Load metadata
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)["metadata"]

            # Load cached processed text blocks
            passages = []
            for pfile in Path("docs/processed").glob("*_clean.json"):
                with open(pfile, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "text" in data:
                        text = data["text"].strip()
                        if text:
                            # split into ~300-word chunks
                            words = text.split()
                            chunk_size = 300
                            chunks = [
                                " ".join(words[i:i + chunk_size])
                                for i in range(0, len(words), chunk_size)
                            ]
                            passages.extend(chunks)
                    elif "sections" in data:
                        for section in data["sections"]:
                            txt = section.get("text", "").strip()
                            if txt:
                                passages.append(txt)

            print(f"✅ Loaded {len(passages)} cached passages.")
            return index, passages, metadata
        else:
            print(f"♻️ Index rebuild triggered by updated files: {', '.join(changed)}")

    # =====================================================
    # 🧠 Build new index (no valid cache found)
    # =====================================================
    print(f"🧠 Scanning folder: {docs_path}")

    supported_exts = [".pdf", ".docx", ".txt"]
    files = [f for f in docs_path.iterdir() if f.suffix.lower() in supported_exts]
    if not files:
        raise FileNotFoundError(f"❌ No supported documents found in {docs_path}")

    print(f"📄 Found {len(files)} documents to process.")

    all_passages = []
    metadata = []  # [(filename, passage_index)]

    # Process each file with Docling
    for f in files:
        try:
            print(f"🧩 Processing {f.name} ...")
            passages = process_with_doclin(str(f))
            all_passages.extend(passages)
            metadata.extend([(f.name, i) for i in range(len(passages))])
        except Exception as e:
            print(f"⚠️ Skipping {f.name}: {e}")

    print(f"✅ Total extracted passages: {len(all_passages)}")

    # =====================================================
    # 🔍 Create DPR embeddings
    # =====================================================
    ctx_model = "facebook/dpr-ctx_encoder-single-nq-base"
    ctx_tokenizer = AutoTokenizer.from_pretrained(ctx_model, use_fast=True)
    ctx_encoder = DPRContextEncoder.from_pretrained(ctx_model)
    ctx_encoder.eval()

    embeddings_list = []
    batch_size = 4

    for i in range(0, len(all_passages), batch_size):
        batch = all_passages[i:i + batch_size]
        inputs = ctx_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = ctx_encoder(**inputs).pooler_output
        embeddings_list.extend(outputs.cpu().numpy())

    embeddings = np.array(embeddings_list, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    # =====================================================
    # 💾 Build and save FAISS index
    # =====================================================
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(index_file))
    print(f"💾 Combined FAISS index saved to: {index_file}")

    # Save metadata mapping
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata}, f, ensure_ascii=False, indent=2)
    print(f"💾 Metadata mapping saved to: {meta_file}")

    return index, all_passages, metadata





# ========================
# Step 3: Passage Retrieval
# ========================
def retrieve_passages(question: str, index, passages: list[str], k: int = 3):
    print(f"🔎 Retrieving top {k} passages for question: {question}")

    Q_MODEL = "facebook/dpr-question_encoder-single-nq-base"
    from transformers import AutoTokenizer
    q_tokenizer = AutoTokenizer.from_pretrained(Q_MODEL, use_fast=True)
    q_encoder = DPRQuestionEncoder.from_pretrained(Q_MODEL)
    q_encoder.eval()

    q_inputs = q_tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        q_emb = q_encoder(**q_inputs).pooler_output.cpu().numpy()

    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)
    retrieved = [passages[i] for i in idxs[0]]

    print(f"✅ Retrieved {len(retrieved)} passages.")

    # optional debug print
    for i, passage in enumerate(retrieved, 1):
        preview = passage[:200] + "..." if len(passage) > 200 else passage
        print(f"\n--- Passage {i} (score: {scores[0][i-1]:.4f}) ---\n{preview}")

    return retrieved



# ========================
# Step 4: Viking-7B Generation
# ========================
def generate_answer(question: str, context_passages: list[str]):
    """
    Generates an answer using Viking-7B.
    - uses a chat-style prompt
    - trims context to avoid the model just echoing it
    - returns only the generated part (not the prompt)
    """
    print("\n⚙️ Generating answer with Viking-7B...")

    model_name = "LumiOpen/Viking-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Make sure we have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Build context but keep it within a safe token budget
    def build_prompt_with_budget(passages, question, max_ctx_tokens=2000):
        system_msg = (
        "Olet asiantunteva tekoälyavustaja ja kirjoitat sujuvaa, täsmällistä suomen kieltä. "
        "Tavoitteesi on antaa perusteltu ja selkeä vastaus käyttäen vain annettua kontekstia. "
        "Jos vastaus ei löydy kontekstista, sano: 'En tiedä varmasti tämän perusteella.' "
        "Kun vastaat:\n"
        "- Käytä luettelomerkkejä, jos asioita on useita.\n"
        "- Korosta tärkeät käsitteet **lihavoimalla**.\n"
        "- Viittaa lähteisiin muodossa [Kappale n].\n"
        "- Älä toista kysymystä tai tyhjiä otsikoita.\n\n"
        )

        user_header = f"Kysymys: {question}\n\nKonteksti:\n"
        ctx = ""
        for i, p in enumerate(passages):
            candidate = ctx + f"[Kappale {i+1}]\n{p}\n\n"
            # arvioi tokenien määrä
            if len(tokenizer.encode(system_msg + user_header + candidate)) > max_ctx_tokens:
                break
            ctx = candidate
        return f"{system_msg}{user_header}{ctx}Vastaus:"


    prompt = build_prompt_with_budget(context_passages, question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1600,
    ).to(model.device)

    print(f"📊 Prompt length: {inputs.input_ids.shape[1]} tokens")

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=700,       # 300 -> 400 to reduce early cut-off
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Keep only the generated part (do not echo context)
    prompt_len = inputs.input_ids.shape[1]
    gen_only = output_ids[0][prompt_len:]
    answer = tokenizer.decode(gen_only, skip_special_tokens=True).strip()

    # If answer is suspiciously short, try to tell the user
    if len(answer) < 15:
        answer = answer + "\n(Huom. vastaus katkesi tai konteksti oli liian pitkä.)"

    print("✅ Generation complete.")
    return answer


# ========================
# Step 5: Main Execution
# ========================
def main():
    """
    Full RAG pipeline across multiple documents in /docs/originals
    """
    print("🚀 Starting multi-document RAG pipeline...")

    base_dir = Path(__file__).resolve().parents[2]
    docs_dir = base_dir / "docs/originals"
    index_path = base_dir / "docs/indexes/combined_index.faiss"

    index, passages, metadata = build_faiss_index_multi(docs_dir, index_path)

    question = "Miten merkkaan lähdeviitteen raportin sisällysluetteloon, jos se on internetlähde?"
    top_passages = retrieve_passages(question, index, passages, k=6)

    answer = generate_answer(question, top_passages)

    print("\n" + "=" * 50)
    print("🎯 FINAL ANSWER FROM VIKING-7B")
    print("=" * 50)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")
    print("\n" + "=" * 50)



import gc
import torch
import faiss

def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
        print("🧹 Cleaned up GPU and memory resources.")

