import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RetrievalAgent:
    def __init__(self, client, embeddings):
        """
        Initializes the Retrieval Agent. It now stores the direct API client.
        """
        print("🔧 Initializing Retrieval Agent...")
        self.client = client # Store the direct API client
        self.hr_retriever = self._create_retriever("dataset/NexaCorp HR Manual.docx", embeddings)
        self.it_retriever = self._create_retriever("dataset/NexaCorp IT Support Manual.docx", embeddings)
        self.payroll_retriever = self._create_retriever("dataset/NexaCorp Payroll Support Manual.docx", embeddings)
        print("✅ Retrieval Agent is ready.")

    def _create_retriever(self, file_path, embeddings):
        """
        Helper function to create a document retriever.
        """
        vectorstore_path = f"faiss_{os.path.basename(file_path)}.index"
        
        if not os.path.exists(vectorstore_path):
            print(f"Creating new vector store for {file_path}...")
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            texts = text_splitter.split_documents(documents)
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(vectorstore_path)
        else:
            print(f"Loading existing vector store for {file_path}...")
            db = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        
        return db.as_retriever()

    def execute_task(self, command: dict) -> str:
        """
        Executes a command by manually performing RAG and calling the Hugging Face API directly.
        """
        domain = command.get('domain')
        query = command.get('query_text')
        
        retriever_map = { "HR": self.hr_retriever, "IT": self.it_retriever, "Payroll": self.payroll_retriever }
        retriever = retriever_map.get(domain)
        
        if not retriever:
            return "Error: Invalid domain specified by Communication Agent."
            
        # 1. Retrieve relevant documents (Updated to fix the deprecation warning)
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. Manually create the prompt for the conversational task
        prompt = f"""Use the following context to answer the question. If you don't know the answer, just say that you don't know. Be concise.
        Context: {context}
        Question: {query}
        """

        # 3. Directly call the API using the correct method for conversational models
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        
        # 4. Return the response using the correct attribute (the fix)
        return response.choices[0].message.content