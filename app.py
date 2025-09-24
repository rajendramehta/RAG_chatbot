import os
import json
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, abort, render_template, Response, session, redirect, url_for
import datetime
import re
import shutil
import logging
from logging.handlers import RotatingFileHandler
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import random
import threading
import gc

# Load environment variables from .env file
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- Configuration & Globals ---
BASE_DIR = Path(__file__).parent.resolve()
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "chroma_index"
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

# --- Global flag for knowledge base status ---
KNOWLEDGE_BASE_READY = False

# --- Database Configuration for both Local and Azure ---
app.config['SECRET_KEY'] = os.environ.get("FLASK_SECRET_KEY", "a-default-secret-key-for-dev")
if 'DB_SERVER' in os.environ:
    print("Connecting to Azure SQL Database...")
    server = os.environ.get("DB_SERVER")
    database = os.environ.get("DB_DATABASE")
    username = os.environ.get("DB_USERNAME")
    password = os.environ.get("DB_PASSWORD")
    connection_string = (f"Driver={{ODBC Driver 17 for SQL Server}};" f"Server=tcp:{server},1433;" f"Database={database};" f"Uid={username};" f"Pwd={password};" "Encrypt=yes;" "TrustServerCertificate=no;" "Connection Timeout=30;")
    app.config['SQLALCHEMY_DATABASE_URI'] = f"mssql+pyodbc:///?odbc_connect={connection_string}"
else:
    print("Connecting to local SQLite Database...")
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + str(BASE_DIR / 'jotbot.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env file.")

# --- Database Models ---
class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class AppLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    level = db.Column(db.String(20), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# --- Logging Configuration ---
log_file = BASE_DIR / 'app.log'
file_handler = RotatingFileHandler(str(log_file), maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('JotBot startup')

# --- LangChain & Model Setup ---
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
streaming_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.1, streaming=True)
gen_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.7, streaming=False)
vector_store = None

# --- Vector Store Functions ---
def build_vector_store():
    print("Building new vector store from PDF and TXT files...")
    pdf_loaders = [PyPDFLoader(str(p)) for p in DOCS_DIR.glob("*.pdf")]
    txt_loaders = [TextLoader(str(p)) for p in DOCS_DIR.glob("*.txt")]
    all_loaders = pdf_loaders + txt_loaders

    if not all_loaders: 
        print("No .pdf or .txt files found in the 'docs' folder.")
        return None
        
    docs = [doc for loader in all_loaders for doc in loader.load()]
    if not docs: 
        print("No documents loaded from files.")
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    index = Chroma.from_documents(split_docs, embeddings_model, persist_directory=str(INDEX_DIR))
    print(f"Index built with {len(split_docs)} chunks.")
    return index

def load_or_build_vector_store(force_rebuild=False):
    global vector_store, KNOWLEDGE_BASE_READY
    
    if vector_store is not None:
        print("Releasing existing vector store from memory...")
        vector_store = None
        gc.collect()

    KNOWLEDGE_BASE_READY = False
    try:
        if force_rebuild and INDEX_DIR.exists():
            print("Forcing rebuild..."); 
            shutil.rmtree(INDEX_DIR)
        
        if not INDEX_DIR.exists():
            vector_store = build_vector_store()
        else:
            print("Loading existing Chroma store..."); 
            vector_store = Chroma(persist_directory=str(INDEX_DIR), embedding_function=embeddings_model)
            print("Store loaded.")
        
        if vector_store is not None:
            KNOWLEDGE_BASE_READY = True
            print("Knowledge base is ready.")
        else:
            print("Knowledge base failed to load.")
    except Exception as e:
        KNOWLEDGE_BASE_READY = False
        app.logger.error(f"Critical error loading vector store: {e}\n{traceback.format_exc()}")
        print(f"Critical error loading vector store: {e}")

def get_vector_store():
    global vector_store
    if not KNOWLEDGE_BASE_READY: return None
    return vector_store

# --- Flask Endpoints ---
@app.route("/")
def index():
    username = request.args.get('username')
    if username:
        session['username'] = username
        return redirect(url_for('index'))
    current_user = session.get('username', 'Guest')
    return render_template("index.html", username=current_user)

@app.route("/status")
def status():
    return jsonify({"ready": KNOWLEDGE_BASE_READY})

@app.route("/refresh_index", methods=["POST"])
def refresh_index_endpoint():
    load_or_build_vector_store(force_rebuild=True)
    if KNOWLEDGE_BASE_READY:
        return jsonify({"status": "success", "message": "Knowledge base has been successfully refreshed."})
    else:
        return jsonify({"status": "error", "message": "Failed to refresh knowledge base."}), 500

@app.route("/suggest_questions", methods=["POST"])
def suggest_questions_endpoint():
    vs = get_vector_store()
    if not vs: return jsonify({"questions": []})
    
    data = request.get_json()
    keyword = data.get("keyword", "").strip()
    if not keyword or len(keyword) < 3: return jsonify({"questions": []})

    try:
        base_retriever = vs.as_retriever(search_type="mmr", search_kwargs={'k': 8, 'fetch_k': 20})
        retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=gen_llm)
        retrieved_docs = retriever.get_relevant_documents(keyword)
        
        if not retrieved_docs: return jsonify({"questions": []})
        
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        prompt_template = """
Your sole task is to generate up to 8 user questions that can be **directly and completely answered** using ONLY the provided text CONTEXT.
- Return ONLY a valid JSON list of strings.
CONTEXT: {context}
JSON Question List: """
        prompt = PromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=gen_llm, prompt=prompt)
        output = chain.run(context=context)
        
        try:
            match = re.search(r'\[.*\]', output, re.DOTALL)
            if match: questions = json.loads(match.group(0))
            else: raise json.JSONDecodeError("No JSON array found", output, 0)
        except json.JSONDecodeError:
            lines = output.strip().split('\n')
            questions = [re.sub(r'^\s*[\d\.\-\*]+\s*', '', line).strip().strip('"\'') for line in lines if len(line) > 10]
        
        random.shuffle(questions)
        return jsonify({"questions": questions[:3]})
    except Exception as e:
        app.logger.error(f"Error in suggest_questions: {e}")
        return jsonify({"questions": []})

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    vs = get_vector_store()
    data = request.get_json(); user_question = data.get("query", "").strip()
    current_user = session.get('username', 'Anonymous')
    
    if not user_question: return Response("Please ask a question.", mimetype='text/plain')
    
    simple_responses = {"hello": "Hello! How can I help you today?", "hi": "Hello! How can I help you today?","hey": "Hello! How can I help you today?","hii": "Hello! How can I help you today?","helo": "Hello! How can I help you today?","quit": "Goodbye! Have a great day.", "exit": "Goodbye! Have a great day.","bye": "Goodbye! Have a great day.", "goodbye": "Goodbye! Have a great day."}
    if user_question.lower() in simple_responses:
        answer = simple_responses[user_question.lower()]
        db.session.add(ChatHistory(username=current_user, question=user_question, answer=answer))
        db.session.commit()
        return Response(answer, mimetype='text/plain')
        
    if not vs: return Response("Knowledge base is not ready. Please try refreshing.", mimetype='text/plain')
    
    try:
        # --- MODIFIED: Implemented the advanced MultiQueryRetriever with MMR ---
        base_retriever = vs.as_retriever(search_type="mmr", search_kwargs={'k': 10, 'fetch_k': 50})
        retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=gen_llm)
        retrieved_docs = retriever.get_relevant_documents(user_question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        answer_prompt = PromptTemplate.from_template("""Answer the user's QUESTION using ONLY the provided CONTEXT. If the answer is not in the context, say "The answer is not in the context."
CONTEXT: {context}
QUESTION: {question}
ANSWER:""")
        answer_chain = LLMChain(llm=streaming_llm, prompt=answer_prompt)
        
        full_answer = ""
        stream_chunks = []
        for chunk in answer_chain.stream({"context": context, "question": user_question}):
            text = chunk.get("text", "")
            full_answer += text
            stream_chunks.append(text)

        history_entry = ChatHistory(username=current_user, question=user_question, answer=full_answer)
        db.session.add(history_entry)
        db.session.commit()
        app.logger.info(f"Saved chat from '{current_user}' to DB.")
        def generate_stream():
            for chunk in stream_chunks:
                yield chunk
        return Response(generate_stream(), mimetype='text/plain')
    except Exception as e:
        error_message = f"Error in /chat: {e}\n{traceback.format_exc()}"
        app.logger.error(error_message)
        db.session.add(AppLog(level='ERROR', message=error_message))
        db.session.commit()
        return Response("An error occurred on the server.", mimetype='text/plain')

@app.route("/history")
def history():
    try:
        all_history = ChatHistory.query.order_by(ChatHistory.timestamp.desc()).all()
        return render_template("history.html", history=all_history)
    except Exception as e:
        app.logger.error(f"Error fetching chat history: {e}")
        return "<h1>Error loading history</h1><p>Could not load chat history from the database.</p>"

@app.route("/logs")
def logs():
    try:
        all_logs = AppLog.query.order_by(AppLog.timestamp.desc()).all()
        return render_template("logs.html", logs=all_logs)
    except Exception as e:
        app.logger.error(f"Error fetching app logs: {e}")
        return "<h1>Error loading logs</h1><p>Could not load app logs from the database.</p>"

# --- Main Execution ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    # Load vector store in a background thread to not block the server from starting
    threading.Thread(target=load_or_build_vector_store).start()
    # app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)