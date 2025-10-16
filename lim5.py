import textwrap
import time
import pdfplumber
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import os
from dotenv import load_dotenv


# ====================================
# SETUP & INITIALIZATION
# ====================================

print("🚀 Starting Nigeria Legal AI Assistant...")

load_dotenv()

# --- API KEYS ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
INDEX_NAME = os.getenv("INDEX_NAME")


# --- CLIENTS ---
print("🧠 Initializing OpenAI and Pinecone clients...")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- CREATE INDEX ---
print("📦 Checking Pinecone index...")
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    print("🆕 Creating Pinecone index...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,  # dimension for text-embedding-3-large
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"✅ Index '{INDEX_NAME}' already exists.")

index = pc.Index(INDEX_NAME)
print("✅ Connected to Pinecone index.\n")


# ====================================
# DOCUMENT EMBEDDING (RUN ONCE)
# ====================================
def embed_documents():
    print("📚 Starting document embedding process...")

    pdf_files = [
        {"path": r"C:\Users\crypt\Downloads\constitution.pdf", "source_type": "constitution"}, #change path
        {"path": r"C:\Users\crypt\Downloads\Tax.pdf", "source_type": "tax_law"} #change path
    ]

    for file in pdf_files:
        pdf_path = file["path"]
        source_type = file["source_type"]
        print(f"\n🔹 Processing document: {pdf_path}")
        print(f"📘 Source type: {source_type}")

        all_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in tqdm(pdf.pages, desc=f"Extracting text from {source_type}"):
                text = page.extract_text()
                if text:
                    all_text += text + "\n\n"

        print(f"📝 Total text length: {len(all_text)} characters")

        # --- Chunking ---
        def chunk_text(text, max_words=1500):
            words = text.split()
            for i in range(0, len(words), max_words):
                yield " ".join(words[i:i + max_words])

        chunks = list(chunk_text(all_text))
        print(f"📑 Created {len(chunks)} chunks for {source_type}")

        # --- Embedding & Upserting ---
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"🔸 Embedding batch {i // batch_size + 1} ({len(batch)} chunks)...")

            embeddings = openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=batch
            ).data

            vectors = [
                {
                    "id": f"{source_type}_{i+j}",
                    "values": embeddings[j].embedding,
                    "metadata": {
                        "text": batch[j],
                        "source_type": source_type,
                        "document_name": pdf_path.split("\\")[-1]
                    }
                }
                for j in range(len(batch))
            ]

            index.upsert(vectors=vectors)
            print(f"✅ Upserted batch {i // batch_size + 1} for {source_type}")
            time.sleep(1)

    print("\n🎉 All documents successfully embedded!\n")

# Uncomment this line ONCE to embed (then comment it again)
# embed_documents()


# ====================================
# RETRIEVAL FUNCTION
# ====================================

def detect_source_type(query):
    """Detect which law applies based on keywords."""
    q = query.lower()
    if "tax" in q or "income" in q or "vat" in q:
        return "tax_law"
    elif "constitution" in q or "right" in q or "chapter" in q:
        return "constitution"
    elif "labour" in q or "employee" in q or "work" in q or "employer" in q:
        return "labour_law"
    else:
        return None

def retrieve_from_pinecone(query, top_k=5):
    print("\n🔍 Step 1: Creating query embedding...")
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    ).data[0].embedding

    source_type = detect_source_type(query)
    if source_type:
        print(f"📄 Detected relevant law: {source_type}")
        filter_dict = {"source_type": source_type}
    else:
        print("⚠️ No specific law detected — searching all sources.")
        filter_dict = {}

    print("🔎 Step 2: Querying Pinecone index...")
    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )

    matches = results["matches"]
    print(f"✅ Retrieved {len(matches)} matching chunks from Pinecone.")
    contexts = [
        f"({m['metadata'].get('source_type','unknown')}) {m['metadata']['text']}"
        for m in matches
    ]
    return contexts


# ====================================
# GPT ANSWER GENERATION
# ====================================

def generate_answer(conversation_prompt, contexts):
    print("\n🧠 Step 3: Generating GPT response with memory...")
    context_text = "\n\n".join(contexts)
    print(f"🧾 Context length: {len(context_text)} characters")

    prompt = f"""
You are an AI Legal Assistant specialized in Nigerian laws, including:
- The Constitution
- The Tax Acts
- The Labour Laws

You remember the ongoing conversation. Continue naturally and maintain context.

If you cannot find an answer in the provided documents, say:
"I couldn’t find an exact match in the provided documents."

Context:
{context_text}

Conversation:
{conversation_prompt}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful Nigerian legal assistant with conversational memory."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=900
    )

    print("✅ GPT response generated successfully!")
    return response.choices[0].message.content



# ====================================
# TELEGRAM BOT HANDLERS
# ====================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("👋 User started the bot.")
    await update.message.reply_text(
        "🧾 Welcome to the Nigeria Legal AI Assistant!\n\n"
        "Ask me about Nigerian laws like the Constitution, Tax Act, or Labour Act.\n\n"
        "Examples:\n"
        "• What does the Constitution say about freedom of speech?\n"
        "• How much tax will I pay if I earn ₦80,000?\n"
        "• Can my employer fire me without notice?"
    )


    # --- Simple in-memory session storage ---
chat_memory = {}
MAX_MEMORY = 5  # store 5 turns per user


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    chat_id = update.effective_chat.id
    print(f"\n💬 New user query from {chat_id}: {user_query}")

    await update.message.reply_text("🔍 Searching legal documents...")

    try:
        # Retrieve previous chat memory (if any)
        conversation_history = chat_memory.get(chat_id, [])
        print(f"🧠 Retrieved {len(conversation_history)} past messages for this user.")

        # Get relevant chunks from Pinecone
        contexts = retrieve_from_pinecone(user_query)

        # Merge conversation memory with current query
        full_prompt = ""
        for turn in conversation_history[-MAX_MEMORY:]:
            full_prompt += f"User: {turn['user']}\nBot: {turn['bot']}\n\n"

        full_prompt += f"User: {user_query}"

        # Generate answer
        answer = generate_answer(full_prompt, contexts)

        # Save this interaction
        conversation_history.append({"user": user_query, "bot": answer})
        chat_memory[chat_id] = conversation_history[-MAX_MEMORY:]

        wrapped = textwrap.fill(answer, width=90)
        await update.message.reply_text(f"🤖 {wrapped}")

    except Exception as e:
        print(f"❌ Error: {e}")
        await update.message.reply_text("⚠️ Sorry, something went wrong.")


# ====================================
# MAIN ENTRY POINT
# ====================================

def main():
    print("🤖 Starting Telegram bot...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Bot is running. Waiting for user messages...")
    app.run_polling()

if __name__ == "__main__":
    main()
