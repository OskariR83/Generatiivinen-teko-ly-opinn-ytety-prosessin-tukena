"""
RAG Pipeline: Doclin + DPR + FAISS + Viking-7B
Project: GENERATIIVINEN TEKOÄLY OPINNÄYTETYÖPROSESSIN TUKENA
Author: <Your Name>
Description:
    End-to-end Retrieval-Augmented Generation (RAG) pipeline that:
    - Uses Doclin for document preprocessing and text cleaning
    - Creates DPR embeddings for retrieval
    - Stores FAISS indexes
    - Generates contextual answers using Viking-7B
"""

import os
import json
import numpy as np
import faiss
import torch
from pathlib import Path
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)
from docling import Docling


# ========================
# Step 1: Doclin Preprocessing
# ========================
def process_with_doclin(file_path):
    """
    Process a document (DOCX/PDF/TXT) with Doclin.
    - Cleans and structures text
    - Saves results to docs/processed/
    - Returns text blocks for embedding
    """
    raw_path = Path(file_path)
    processed_dir = Path("docs/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_file = processed_dir / f"{raw_path.stem}_clean.json"

    # Use cached processed file if available
    if output_file.exists():
        print(f"📂 Using cached Doclin output: {output_file}")
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print(f"🧠 Processing document with Doclin: {file_path}")
        model = Doclin.from_pretrained("LumiOpen/Doclin-base")
        data = model.process(file_path)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Cleaned document saved to: {output_file}")

    # Extract text blocks
    text_blocks = []
    if "sections" in data:
        for section in data["sections"]:
            text = section.get("text", "").strip()
            if text:
                text_blocks.append(text)
    elif "text" in data:
        text_blocks.append(data["text"].strip())

    print(f"✅ Extracted {len(text_blocks)} text blocks from Doclin output.")
    return text_blocks


# ========================
# Step 2: Build FAISS Index
# ========================
def build_faiss_index(doc_path, index_path=None):
    """
    Encode document passages and build a FAISS index.
    Saves index under docs/indexes/
    """
    passages = process_with_doclin(doc_path)
    if not passages:
        raise ValueError("❌ No text passages extracted by Doclin.")

    index_dir = Path("docs/indexes")
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_path or index_dir / f"{Path(doc_path).stem}.faiss"

    print(f"🔍 Creating DPR embeddings for {len(passages)} passages...")

    ctx_model = "facebook/dpr-ctx_encoder-single-nq-base"
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_model)
    ctx_encoder = DPRContextEncoder.from_pretrained(ctx_model)

    embeddings_list = []
    batch_size = 4

    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        inputs = ctx_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = ctx_encoder(**inputs).pooler_output
        embeddings_list.extend(outputs.cpu().numpy())

    embeddings = np.array(embeddings_list, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_path))

    print(f"✅ FAISS index built and saved to: {index_path}")
    return index, passages


# ========================
# Step 3: Passage Retrieval
# ========================
def retrieve_passages(question, index, passages, k=3):
    """
    Encode a question and retrieve top-k relevant passages using FAISS.
    """
    print(f"🔎 Retrieving top {k} passages for question: {question}")

    q_encoder_model = "facebook/dpr-question_encoder-single-nq-base"
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(q_encoder_model)
    q_encoder = DPRQuestionEncoder.from_pretrained(q_encoder_model)

    inputs = q_tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        q_emb = q_encoder(**inputs).pooler_output.cpu().numpy()

    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)
    retrieved = [passages[i] for i in idxs[0]]

    print(f"✅ Retrieved {len(retrieved)} passages.")
    return retrieved


# ========================
# Step 4: Viking-7B Generation
# ========================
def generate_answer(question, context_passages):
    """
    Generate an answer using Viking-7B based on retrieved context.
    """
    print("⚙️ Generating answer with Viking-7B...")

    model_name = "LumiOpen/Viking-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    context_text = "\n\n".join(context_passages)
    prompt = (
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_text}\n\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("✅ Generation complete.")
    return answer


# ========================
# Step 5: Main Execution
# ========================
def main():
    """
    Run the full Doclin → DPR → FAISS → Viking pipeline.
    """
    doc_path = "docs/originals/example.docx"
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"❌ Document not found: {doc_path}")

    index, passages = build_faiss_index(doc_path)

    question = "Mitä tekoälytyökaluja dokumentissa mainitaan?"

    top_passages = retrieve_passages(question, index, passages, k=3)
    answer = generate_answer(question, top_passages)

    print("\n===============================")
    print("🎯 Final Answer from Viking-7B")
    print("===============================\n")
    print(answer)


if __name__ == "__main__":
    main()
