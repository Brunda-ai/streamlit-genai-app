# rag_engine.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import prompts

class RAGEngine:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings

    def ingest_documents(self, pdf_paths: list) -> FAISS:
        """Processes document inputs, computes chunk frames, and fills a local FAISS index."""
        all_docs = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            all_docs.extend(loader.load())
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        vector_store = FAISS.from_documents(splits, self.embeddings)
        return vector_store

    def execute_rag_pipeline(self, query: str, history: list, vector_store: FAISS, intent: str) -> dict:
        """Orchestrates an LCEL RAG pipeline resilient across LangChain v0.2.x and v0.3.x."""
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # 1. Contextualize Question Stage
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("system", prompts.CONTEXTUALIZE_QUESTION_INSTRUCTION)
        ])
        
        # If there is history, condense the question; otherwise pass it through
        if history:
            condense_chain = contextualize_q_prompt | self.llm | StrOutputParser()
        else:
            condense_chain = RunnablePassthrough() | (lambda x: x["input"])

        # 2. Extract & Validate Documents Programmatically (Guardrail Interceptor)
        input_data = {"input": query, "chat_history": history}
        retrieved_docs_chain = condense_chain | retriever
        raw_context_docs = retrieved_docs_chain.invoke(input_data)
        
        # Clean and stringify document contents to check if information is actually present
        combined_text = "".join(doc.page_content.strip() for doc in raw_context_docs)
        
        # CRITICAL CHECK: If no chunks were returned or the data inside them is blank
        if not raw_context_docs or len(combined_text) == 0:
            return {
                "answer": "Product information not available, please upload your propriety documents",
                "context": []
            }

        # 3. QA Generation Stage (Proceeds only if context documents exist)
        selected_template = prompts.RESPONSE_TEMPLATES.get(intent, prompts.RESPONSE_TEMPLATES["General Knowledge"])
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", selected_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Helper function to format list of documents to a single string
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 4. Constructing and Executing the Final Generation Sequence
        lcel_rag_chain = (
            RunnablePassthrough.assign(context=retrieved_docs_chain)
            | RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | qa_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        generated_answer = lcel_rag_chain.invoke(input_data)
        
        return {
            "answer": generated_answer,
            "context": raw_context_docs
        }