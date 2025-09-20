# Step 1: Install required libraries
!pip install -q transformers pdfplumber

# Step 2: Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdfplumber
import torch
from google.colab import files

# Step 3: Upload your PDF file
uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]

# Step 4: Extract text from the uploaded PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
    return text

pdf_text = extract_text_from_pdf(pdf_path)
print("✅ Extracted text length:", len(pdf_text))
print("\n📄 Preview:\n", pdf_text[:500])

# Step 5: Load Granite model and tokenizer
print("\n🔁 Loading Granite model...")
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.3-2b-instruct")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("✅ Model loaded on", device)

# Step 6: Define Q&A function using the Granite model
def ask_granite(question, context, max_tokens=150):
    messages = [
        {"role": "system", "content": "You are an AI assistant that answers questions based on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return answer.strip()

# Step 7: First question example
question = "Summarize the main topic of this document."
answer = ask_granite(question, pdf_text[:3000])  # Truncate to 3000 chars for context
print(f"\n📌 Q: {question}\n🧠 A: {answer}")

# Step 8: Interactive Q&A loop
print("\n🔎 Ask questions about the PDF. Type 'exit' to quit.")
while True:
    try:
        q = input("Your question: ")
        if q.lower().strip() == "exit":
            print("👋 Exiting.")
            break
        a = ask_granite(q, pdf_text[:3000])  # Optional: chunk or truncate
        print(f"🧠 Answer: {a}\n")
    except Exception as e:
        print(f"❌ Error: {e}")
