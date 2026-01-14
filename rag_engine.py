import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        
        # Expert Persona
        self.llm = ChatGroq(
            temperature=0.1, 
            model_name="llama-3.3-70b-versatile", 
            api_key=os.getenv("GROQ_API_KEY")
        )

    def validate_is_resume(self, docs):
        full_text = " ".join([d.page_content.lower() for d in docs[:2]])
        keywords = ["education", "experience", "skills", "projects", "summary", "contact", "b.tech", "university", "work"]
        hit_count = sum(1 for k in keywords if k in full_text)
        return hit_count >= 3

    def ingest_file(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        if not self.validate_is_resume(docs):
            raise ValueError("Document rejected: Does not appear to be a Resume/CV.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
        else:
            self.vector_store.add_documents(splits)
            
        return len(splits)

    def get_chain(self):
        """Returns the appropriate chain based on whether a file exists."""
        
        # --- SCENARIO 1: NO RESUME UPLOADED (General Advice Mode) ---
        if self.vector_store is None:
            system_prompt = (
                "You are an expert Senior Technical Recruiter and Career Coach."
                "The user has NOT uploaded a resume yet. "
                "Answer their general questions about resume building, or resume related tips, and STRICTLY prohibit to answer or give advice or any suggestions for a query that asks for information which is out of the scope of general resume based context."
                "Be concise (max 3 sentences) unless asked for details. "
                "STRICTLY prohibit to answer any query that asks for information which is out of the scope of general resume based career advice."
                "Do NOT make up facts about a specific user resume since none is uploaded."
                
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            # Simple Chain: Prompt -> LLM -> String Output
            return prompt | self.llm | StrOutputParser()

        # --- SCENARIO 2: RESUME UPLOADED (RAG Mode) ---
        else:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            
            system_prompt = (
                "You are an expert Senior Technical Recruiter. "
                "Use the provided resume context to answer the user's questions, and STRICTLY prohibit to answer or give advice or any suggestions for a query that asks for information which is out of the scope of general resume based context or the provided resume."
                "STRICTLY prohibit to answer any query that asks for information which is out of the scope of general resume based career advice or the provided resume."
                "Instructions:"
                "Be concise (3-4 sentences max) unless asked for details."
                "\n\n"
                "Context:\n{context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            return rag_chain