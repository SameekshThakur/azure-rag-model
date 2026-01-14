import chainlit as cl
from rag_engine import RAGEngine
import os

@cl.on_chat_start
async def start():
    """Runs when a user connects."""
    session_engine = RAGEngine()
    cl.user_session.set("engine", session_engine)
    
    await cl.Message(content="""# 🚀 AI Resume Architect

### Your Intelligent Career Consultant

Welcome! I am an advanced RAG-powered assistant designed to analyze resumes and provide career guidance.

**Capabilities:**

- 📄 **Deep Resume Analysis:** Upload your PDF to get critiques, find gaps, and extract key details.
- 🎯 **Context-Aware:** I know when to use your resume and when to just chat generally.
- 🔒 **Secure & Private:** Your data is processed in-memory and wiped when you leave.

---

### 💡 How to get started:

1.  **Upload your Resume** (PDF format) using the attachment icon.
2.  Ask specific questions like _"What are my strengths?"_
3.  Or ask general questions like _"How do I prepare for a System Design interview?"_
""").send()

@cl.on_message
async def main(message: cl.Message):
    engine = cl.user_session.get("engine")
    
    # 1. Handle File Upload
    if message.elements:
        processing_msg = cl.Message(content=f"🔍 Analyzing document...")
        await processing_msg.send()
        
        try:
            pdf_file = message.elements[0]
            num_chunks = await cl.make_async(engine.ingest_file)(pdf_file.path)
            processing_msg.content = f"✅ **Resume Accepted!**"
            await processing_msg.update()
        except ValueError as e:
            processing_msg.content = f"❌ **Error:** {str(e)}"
            await processing_msg.update()
            return 
        except Exception as e:
            processing_msg.content = f"❌ **System Error:** {str(e)}"
            await processing_msg.update()
            return

    user_query = message.content
    if message.elements and not user_query:
        return

    # 2. Get the Adaptive Chain
    chain = engine.get_chain()
    
    msg = cl.Message(content="")

    # 3. Execute Chain
    # We use stream=False usually for RAG, but let's just invoke it.
    res = await chain.ainvoke({"input": user_query}, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # 4. Handle Response Types
    # If res is a String, it means we used the "General Advice" chain (No Sources)
    if isinstance(res, str):
        msg.content = res
        await msg.send()
        
    # If res is a Dict, it means we used the "RAG" chain (Has Sources)
    elif isinstance(res, dict):
        answer = res["answer"]
        source_documents = res.get("context", [])

        # Restore Clickable Source Links
        text_elements = []
        if source_documents:
            for idx, doc in enumerate(source_documents):
                source_name = f"Source {idx+1}"
                text_elements.append(
                    cl.Text(content=doc.page_content, name=source_name, display="side")
                )
            
            answer += "\n\n**Sources:**"
            for idx in range(len(source_documents)):
                answer += f" *Source {idx+1}* "

        msg.content = answer
        msg.elements = text_elements
        await msg.send()