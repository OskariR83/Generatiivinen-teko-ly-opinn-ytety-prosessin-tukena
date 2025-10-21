"""
RAG Pipeline: Doclin + DPR + FAISS + Viking-7B
Project: GENERATIIVINEN TEKOÄLY OPINNÄYTETYÖPROSESSIN TUKENA
Author: <Your Name>
Description:
    This pipeline processes a document with Doclin, encodes text with DPR,
    retrieves relevant passages with FAISS, and uses Viking-7B to generate
    final answers.
"""

import os
import numpy as np
import faiss
import torch
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)
from doclin import Doclin


# ========================
# Step 1: Preprocess with Doclin
# ========================
def process_with_doclin(file_path):
    """
    Use Doclin to read and structure a document (DOCX, PDF, TXT).
    Returns a list of clean text segments.
    """
    print(f"🧠 Processing document with Doclin: {file_path}")
    model = Doclin.from_pretrained("LumiOpen/Doclin-base")

    # Doclin handles reading & text extraction automatically
    results = model.process(file_path)

    text_blocks = []
    if "sections" in results:
        for section in results["sections"]:
            text = section.get("text", "").strip()
            if text:
                text_blocks.append(text)
    elif "text" in results:
        text_blocks.append(results["text"].strip())

    print(f"✅ Extracted {len(text_blocks)} text blocks with Doclin.")
    return text_blocks


# ========================
# Step 2: Build FAISS index
# ========================
def build_faiss_index(doc_path, index_path="my_index.faiss"):
    """
    Build a FAISS index from Doclin-processed document passages.
    """
    passages = process_with_doclin(doc_path)
    if not passages:
        raise ValueError("No text passages extracted by Doclin.")

    print(f"🔍 Creating embeddings for {len(passages)} passages...")

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
    faiss.write_index(index, index_path)

    print(f"✅ FAISS index built and saved to {index_path}")
    return index, passages


# ========================
# Step 3: Retrieve relevant passages
# ========================
def retrieve_passages(question, index, passages, k=3):
    """
    Encode a question and retrieve top-k most relevant passages using FAISS.
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
# Step 4: Generate final answer with Viking-7B
# ========================
def generate_answer(question, context_passages):
    """
    Combine retrieved passages with the question and generate an answer
    using LumiOpen/Viking-7B.
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
        "You are a helpful assistant. Use the following context to answer the question in Finnish language.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_text}\n\nAnswer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

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
# Step 5: Full pipeline
# ========================
def main():
    """
    Run the full RAG pipeline on a document using Doclin + DPR + FAISS + Viking-7B.
    """
    doc_path = "docs/example.docx"

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Document not found: {doc_path}")

    # 1. Build FAISS index using Doclin-extracted text
    index, passages = build_faiss_index(doc_path)

    # 2. Ask a question
    question = "Mitä tekoälytyökaluja dokumentissa mainitaan?"

    # 3. Retrieve context and generate an answer
    top_passages = retrieve_passages(question, index, passages, k=3)
    answer = generate_answer(question, top_passages)

    print("\n===============================")
    print("🎯 Final Answer from Viking-7B:")
    print("===============================\n")
    print(answer)


if __name__ == "__main__":
    main()
