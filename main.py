import nltk
from fastapi import FastAPI, HTTPException,UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, BartTokenizer, BartForConditionalGeneration
from typing import Literal
import uuid
import PyPDF2


try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    raise Exception(f"NLTK veri setleri yüklenirken hata: {e}")


model_name = "facebook/mbart-large-50-many-to-many-mmt"
summarization_model = "facebook/bart-large-cnn"

try:
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
except Exception as e:
    raise Exception(f"Çeviri modeli yüklenirken hata: {e}")

try:
    summarizer_tokenizer = BartTokenizer.from_pretrained(summarization_model)
    summarizer = BartForConditionalGeneration.from_pretrained(summarization_model)
except Exception as e:
    raise Exception(f"Özetleme modeli yüklenirken hata: {e}")


app = FastAPI()

# İstek modeli
class SummarizeRequest(BaseModel):
    text: str
    summary_type: Literal["detailed", "normal", "short"] = "normal"

def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file.file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF metin çıkarma başarısız: {str(e)}")

# Özet uzunluğunu hesapla
def calculate_text_length(text):
    return len(text.split())

def calculate_summary_length(text_length, summary_type="normal"):
    if summary_type == "detailed":
        if text_length > 1000:
            return int(text_length * 0.40)
        elif text_length > 100:
            return int(text_length * 0.35)
        else:
            return int(text_length * 0.30)
    elif summary_type == "normal":
        if text_length > 1000:
            return int(text_length * 0.25)
        elif text_length > 100:
            return int(text_length * 0.20)
        else:
            return int(text_length * 0.15)
    elif summary_type == "short":
        if text_length > 1000:
            return int(text_length * 0.15)
        elif text_length > 100:
            return int(text_length * 0.10)
        else:
            return int(text_length * 0.05)
    else:
        raise ValueError("Geçersiz özet türü")

# Metni kayan pencere ile parçala
def split_text_by_sentences_with_sliding_window(text, max_tokens=500, overlap_ratio=0.2):
    try:
        sentences = nltk.sent_tokenize(text, language='turkish')
    except Exception as e:
        print(f"Cümle tokenizasyonu sırasında hata: {e}")
        return []

    chunks = []
    current_chunk = []
    current_token_count = 0
    sentence_index = 0
    overlap_tokens = int(max_tokens * overlap_ratio)

    while sentence_index < len(sentences):
        sentence = sentences[sentence_index]
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))

        if current_token_count + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_token_count += sentence_tokens
            sentence_index += 1
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                overlap_sentence_count = max(1, int(len(current_chunk) * overlap_ratio))
                sentence_index -= overlap_sentence_count
                current_chunk = sentences[sentence_index:sentence_index + overlap_sentence_count]
                current_token_count = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk)
                sentence_index += overlap_sentence_count
            else:
                chunks.append(sentence)
                sentence_index += 1
                current_chunk = []
                current_token_count = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Dinamik özetleme
def summarize_dynamic(text, tokenizer, model, summary_type="normal"):
    text_length = calculate_text_length(text)
    summary_length = calculate_summary_length(text_length, summary_type)
    
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max(summary_length + 50, 30),
        min_length=max(summary_length - 20, 10),
        num_beams=4 if summary_type == "detailed" else 2,
        length_penalty=1.0 if summary_type != "short" else 0.8,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Çeviri
def translate(text, tokenizer, model, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_ids = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=512
    )
    return tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]

# Özetleri birleştir
def merge_summaries(summaries, summary_type="normal"):
    merged = []
    for i, summary in enumerate(summaries):
        if i == 0:
            merged.append(summary)
        else:
            prev_summary = merged[-1]
            if not summary.strip() or summary in prev_summary:
                continue
            if summary_type == "short":
                if len(summary.split()) > len(prev_summary.split()) * 0.5:
                    merged.append(summary)
            else:
                merged.append(summary)
    return " ".join(merged)

#özetleme
def process_text(text, summary_type="normal"):
    if not text.strip():
        raise ValueError("Boş metin girildi")

    print("/*\/*\ Metin cümlelere göre parçalara ayrılıyor...")
    chunks = split_text_by_sentences_with_sliding_window(text, max_tokens=500, overlap_ratio=0.2)
    if not chunks:
        raise ValueError("Metin parçalara ayrılamadı")
    print(f"/*\/*\ Metin {len(chunks)} parçaya ayrıldı.\n")

    final_summaries_tr = []

    for i, chunk in enumerate(chunks, 1):
        print(f"/*\/*\ Parça {i} işleniyor (Token sayısı: {len(tokenizer.encode(chunk, add_special_tokens=False))})...")

        try:
            print("/*\/*\ Parça İngilizceye çevriliyor...")
            chunk_en = translate(chunk, tokenizer, model, src_lang="tr_TR", tgt_lang="en_XX")
            print(f"/*\/*\ İngilizce çeviri: {chunk_en}\n")
        except Exception as e:
            print(f"Çeviri hatası (Parça {i}): {e}")
            continue

        try:
            print("/*\/*\ İngilizce parça özetleniyor...")
            summary_en = summarize_dynamic(chunk_en, summarizer_tokenizer, summarizer, summary_type)
            print(f"/*\/*\ İngilizce özet: {summary_en}\n")
        except Exception as e:
            print(f"Özetleme hatası (Parça {i}): {e}")
            continue

        try:
            print("/*\/*\ Özet Türkçeye çevriliyor...")
            summary_tr = translate(summary_en, tokenizer, model, src_lang="en_XX", tgt_lang="tr_TR")
            print(f"/*\/*\ Türkçe özet: {summary_tr}\n")
        except Exception as e:
            print(f"Geri çeviri hatası (Parça {i}): {e}")
            continue

        final_summaries_tr.append(summary_tr)

    final_summary = merge_summaries(final_summaries_tr, summary_type)
    print(f"/*\/*\ Tüm parçalar işlendi. Nihai {summary_type} Türkçe özet:")
    print(final_summary)
    return final_summary


@app.post("/summarize", response_model=dict)
async def summarize_text(request: SummarizeRequest):
    try:
        summary = process_text(request.text, request.summary_type)
        return {"summary": summary}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Özetleme sırasında hata: {str(e)}")
@app.post("/pdfsummarize", response_model=dict)
async def extract_pdf_text(pdfFile: UploadFile = File(...)):

    if not pdfFile.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Sadece PDF dosyaları kabul edilir")
    

    text = extract_text_from_pdf(pdfFile)
    summary = process_text(text, "normal")
    return {"summary": summary}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)