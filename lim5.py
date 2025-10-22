import textwrap
import time
import pdfplumber
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv


# ====================================
# 1️⃣ SETUP & INITIALIZATION
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
# 2️⃣ DOCUMENT EMBEDDING (RUN ONCE)
# ====================================
def embed_documents():
    print("📚 Starting document embedding process...")

    pdf_files = [
        {"path": r"C:\Users\crypt\Downloads\Traffic.pdf", "source_type": "Traffic Law"},
    ]

    for file in pdf_files:
        pdf_path = file["path"]
        source_type = file["source_type"]
        print(f"\n🔹 Processing document: {pdf_path}")
        print(f"📘 Source type: {source_type}")

        # --- Semantic Chunking with Table Preservation ---
        chunks = []
        total_tables = 0
        total_text_chunks = 0
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc=f"Processing {source_type} pages")):
                # Extract tables as complete units
                tables = page.extract_tables()
                for i, table in enumerate(tables):
                    if table and any(any(cell is not None for cell in row) for row in table):
                        table_chunk = f"TABLE Page {page_num+1}:\n"
                        for row in table:
                            # Filter out None values and convert to strings
                            row_data = [str(cell) if cell is not None else "" for cell in row]
                            table_chunk += " | ".join(row_data) + "\n"
                        chunks.append(table_chunk)
                        total_tables += 1
                
                # Extract and chunk text semantically
                text = page.extract_text()
                if text and text.strip():
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100,
                        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
                    )
                    text_chunks = text_splitter.split_text(text)
                    chunks.extend(text_chunks)
                    total_text_chunks += len(text_chunks)

        print(f"📝 Created {len(chunks)} total chunks for {source_type}")
        print(f"📊 Breakdown: {total_tables} table chunks + {total_text_chunks} text chunks")

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
# 3️⃣ GREETING DETECTION (NO API CALL)
# ====================================

def is_greeting(message: str) -> bool:
    """
    Detect if the message is a greeting without needing AI
    """
    message = message.strip().lower()
    
    # Common greetings
    simple_greetings = [
        'hello', 'hi', 'hey', 'greetings', 'howdy',
        'good morning', 'good afternoon', 'good evening',
        'morning', 'afternoon', 'evening',
        'hi there', 'hello there', 'hey there',
        'sup', 'whats up', "what's up", 'yo',
        'hiya', 'heya', 'hola'
    ]
    
    # Check exact matches
    if message in simple_greetings:
        return True
    
    # Check if message starts with greeting (e.g., "hi, how are you?")
    for greeting in simple_greetings:
        if message.startswith(greeting + ' ') or message.startswith(greeting + ','):
            return True
    
    return False


def get_greeting_response() -> str:
    """
    Return a friendly greeting response
    """
    return (
        "👋 *Hello! Welcome to the Nigerian Legal AI Assistant*\n\n"
        "I can help you with questions about:\n"
        "📜 *The Constitution*\n"
        "💰 *Tax Acts*\n"
        "🚗 *Traffic Laws*\n\n"
        "*Examples:*\n"
        "• What does the Constitution say about freedom of speech?\n"
        "• How is income tax calculated?\n"
        "• What is the penalty for speeding?\n\n"
        "💬 How can I assist you today?"
    )


# ====================================
# 4️⃣ RETRIEVAL FUNCTION
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
    elif "traffic" in q or "road" in q or "driving" in q or "vehicle" in q:
        return "Traffic Law"
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
# 5️⃣ GPT ANSWER GENERATION
# ====================================

def generate_answer(conversation_prompt, contexts, user_input):
    print("\n🧠 Step 3: Generating GPT response with memory...")
    context_text = "\n\n".join(contexts)
    print(f"🧾 Context length: {len(context_text)} characters")

    prompt = f"""
# SYSTEM PROMPT
You are an AI Legal Assistant specialized in Nigerian laws, including:
- The Constitution
- The Tax Acts
- The Traffic Laws

You remember the ongoing conversation. Continue naturally and maintain context.
If you cannot find an answer in the provided documents, say:
"I couldn't find an exact match in the provided documents."

IMPORTANT: Always format your responses clearly:
- Use clear paragraphs
- Use bullet points when listing multiple items
- Bold important terms using *text* for Telegram markdown
- Always cite sources at the end with source type, section numbers, and page numbers

# CONTEXT
{context_text}

# CONVERSATION HISTORY
{conversation_prompt}

# USER PROMPT
{user_input}

# EXAMPLES
Example 1:
User: "What is the penalty for speeding?"
AI: "According to the Traffic Law, the penalty for speeding varies based on the offense:

- *First-time offenders*: Fine of ₦10,000
- *Repeat offenders*: Fine of ₦20,000 or imprisonment

📚 *Source:* Traffic Law, Section 15, Page 23"

Example 2:
User: "What are my constitutional rights?"
AI: "The Nigerian Constitution guarantees several *fundamental rights* in Chapter IV, including:

- *Right to life* (Section 33)
- *Right to dignity* (Section 34)
- *Right to fair hearing* (Section 36)
- *Freedom of expression* (Section 39)

These rights protect citizens from unlawful treatment and ensure justice.

📚 *Source:* Constitution of Nigeria, Chapter IV, Sections 33-46"

Example 3:
User: "How is income tax calculated?"
AI: "Income tax in Nigeria is calculated using *progressive tax rates* based on your annual income:

- First ₦300,000: 7%
- Next ₦300,000: 11%
- Next ₦500,000: 15%
- Next ₦500,000: 19%
- Above ₦1,600,000: 24%

For example, if you earn ₦80,000 monthly (₦960,000 annually), your tax would be calculated across multiple brackets.

📚 *Source:* Personal Income Tax Act, Section 33, Page 45"

Example 4:
User: "What about space law?"
AI: "I couldn't find an exact match in the provided documents about space law. I specialize in Nigerian *constitutional law*, *tax law*, and *traffic law*.

Is there anything related to these areas I can help you with?"
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful Nigerian legal assistant with conversational memory. Always format responses clearly with markdown."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=900
    )

    print("✅ GPT response generated successfully!")
    return response.choices[0].message.content


# ====================================
# 6️⃣ TELEGRAM BOT HANDLERS
# ====================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("👋 User started the bot.")
    chat_id = update.effective_chat.id
    
    # Clear chat memory on start
    if chat_id in chat_memory:
        chat_memory[chat_id] = []
    
    welcome_message = (
        "🧾 *Welcome to the Nigerian Legal AI Assistant!*\n\n"
        "I can help you understand Nigerian laws including:\n"
        "📜 The Constitution\n"
        "💰 Tax Acts\n"
        "🚗 Traffic Laws\n\n"
        "*Examples:*\n"
        "• What does the Constitution say about freedom of speech?\n"
        "• How much tax will I pay if I earn ₦80,000?\n"
        "• What is the penalty for overspeeding?\n\n"
        "💬 Ask me anything about Nigerian laws!\n"
        "🔄 Use /clear to reset our conversation"
    )
    
    await update.message.reply_text(welcome_message, parse_mode='Markdown')


async def clear_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear conversation history"""
    chat_id = update.effective_chat.id
    if chat_id in chat_memory:
        chat_memory[chat_id] = []
    print(f"🔄 Cleared chat memory for user {chat_id}")
    await update.message.reply_text(
        "✅ *Conversation history cleared!*\n\nYou can start a fresh conversation now.",
        parse_mode='Markdown'
    )


# --- Simple in-memory session storage ---
chat_memory = {}
MAX_MEMORY = 5  # store 5 turns per user


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    chat_id = update.effective_chat.id
    print(f"\n💬 New user query from {chat_id}: {user_query}")

    # ✅ CHECK FOR GREETINGS FIRST (NO API CALL)
    if is_greeting(user_query):
        greeting_response = get_greeting_response()
        await update.message.reply_text(greeting_response, parse_mode='Markdown')
        print("👋 Greeting detected - responded without API call")
        return

    # Show typing indicator
    await update.message.chat.send_action(action="typing")
    await update.message.reply_text("🔍 *Searching legal documents...*", parse_mode='Markdown')

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
        answer = generate_answer(full_prompt, contexts, user_query)

        # Save this interaction
        conversation_history.append({"user": user_query, "bot": answer})
        chat_memory[chat_id] = conversation_history[-MAX_MEMORY:]

        # Send formatted response
        await update.message.reply_text(answer, parse_mode='Markdown')
        print("✅ Response sent successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        await update.message.reply_text(
            "⚠️ *Sorry, something went wrong.*\n\nPlease try again or rephrase your question.",
            parse_mode='Markdown'
        )


# ====================================
# 7️⃣ MAIN ENTRY POINT
# ====================================

def main():
    print("🤖 Starting Telegram bot...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear_chat))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Bot is running. Waiting for user messages...")
    app.run_polling()

if __name__ == "__main__":
    main()