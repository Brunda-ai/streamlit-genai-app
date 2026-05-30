import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Import custom modular framework dependencies
import utils
from intent_router import IntentRouter
from rag_engine import RAGEngine

# Automatically load verified API keys from the local environment configurations
#load_dotenv()

st.set_page_config(page_title="Smart Support Copilot", page_icon="🤖", layout="wide")

# Initialize foundation infrastructure variables once per session context
if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(
        api_key=st.secrets["AZURE_OPENAI_API_KEY"], # Fallback to st.secrets or os.environ
        base_url=st.secrets["AZURE_OPENAI_BASE_URL"],
        model='gpt-4o-mini',
        temperature=0.3
    )
if "embeddings" not in st.session_state:
    st.session_state.embeddings = OpenAIEmbeddings(
        api_key=st.secrets.get("AZURE_OPENAI_API_KEY"),
        base_url=st.secrets.get("AZURE_OPENAI_BASE_URL"),
        model='text-embedding-3-small'
    )

# Bootstrap runtime engines
if "router" not in st.session_state:
    st.session_state.router = IntentRouter(st.session_state.llm)
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine(st.session_state.llm, st.session_state.embeddings)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🤖 Smart Support Copilot")
st.caption("Fully Decoupled Enterprise Architecture with Isolated Utilities Layer")

# --- Sidebar Ingestion UI Block ---
with st.sidebar:
    st.header("📂 Ingestion Management")
    uploaded_files = st.file_uploader(
        "Upload Technical Docs / Manuals / System FAQs", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Rebuild Core Vector Index", type="primary") and uploaded_files:
        with st.spinner("Processing documents into vector frames..."):
            
            # Use utility functions to save binary streams to disk safely
            saved_paths = utils.save_uploaded_files(uploaded_files)
            
            # Execute storage vector processing via the RAG engine component
            st.session_state.vector_store = st.session_state.rag.ingest_documents(saved_paths)
            
            # Purge footprints safely using utility cleanup loop
            utils.cleanup_local_paths(saved_paths)
            st.success("Vector framework successfully synchronized!")

# --- Operational Chat Screen Processing Structure ---
if not st.session_state.vector_store:
    st.info("👈 Please populate your knowledge index in the sidebar to activate processing capabilities.")
else:
    # Print clean historical conversation components
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Capturing input requests
    if user_input := st.chat_input("Enter customer query or system troubleshooting issue..."):
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # Format session tracking arrays into LangChain schemas via utilities
        formatted_history = utils.format_chat_history(st.session_state.chat_history)
        
        # 1. Processing Categorization/Routing Decisions via decoupled module
        with st.spinner("Classifying Query Intent..."):
            detected_intent = st.session_state.router.classify_intent(user_input, formatted_history)
            
        with st.chat_message("assistant"):
            st.caption(f"🎯 **Identified Intent Strategy:** `{detected_intent}`")
            
            # 2. Dynamic execution and answer production via the RAG engine module
            with st.spinner("Analyzing knowledge bases & rendering solution payload..."):
                output_payload = st.session_state.rag.execute_rag_pipeline(
                    query=user_input,
                    history=formatted_history,
                    vector_store=st.session_state.vector_store,
                    intent=detected_intent
                )
                
                generated_answer = output_payload["answer"]
                source_documents = output_payload.get("context", [])
                
                st.markdown(generated_answer)
                
                # 3. Source Traceability Panel
                with st.expander("📝 Traceability & Grounding Verification"):
                    if source_documents:
                        st.markdown("**System Grounding:** Fully anchored via verified local reference data.")
                        for idx, document in enumerate(source_documents):
                            # Parse document metadata details safely via utils layer
                            origin_file, page_offset = utils.parse_source_metadata(document)
                            st.markdown(f"**Doc Ref {idx+1}:** `{origin_file}` (Page {page_offset})")
                            st.caption(f"*Text excerpt snippet:* ...{document.page_content[:140].strip()}...")
                    else:
                        st.markdown("⚠️ **System Grounding:** Contextual mismatch detected; response generated using global model logic.")
                        
        # Append parameters to reactive application state
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": generated_answer})