# Step 1: Install required libraries
!pip install transformers pdfplumber

# Step 2: Import libraries
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pdfplumber
import torch

# Step 3: Upload your PDF file
from google.colab import files
uploaded = files.upload()

# Step 4: Extract text from the uploaded PDF
pdf_path = list(uploaded.keys())[0]

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

pdf_text = extract_text_from_pdf(pdf_path)
print("Extracted text length:", len(pdf_text))

# Optional: Print first 500 characters of PDF text
print(pdf_text[:500])

# Step 5: Load Granite model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Step 6: Define Q&A function using the granite model
def ask_granite(question, context, max_tokens=100):
    # Prepare messages with context + question
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
    # Decode generated tokens, skipping input tokens
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return answer.strip()

# Step 7: Example usage - ask a question based on the PDF
question = "Summarize the main topic of this document."
answer = ask_granite(question, pdf_text[:3000])  # Limit context to first 3000 chars for speed

print(f"Q: {question}\nA: {answer}")

# Step 8: Interactive question-answer loop
print("\nAsk questions about the PDF. Type 'exit' to quit.")
while True:
    q = input("Your question: ")
    if q.lower() == "exit":
        break
    a = ask_granite(q, pdf_text[:3000])
    print(f"Answer: {a}\n")
