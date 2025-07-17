import os
import sys
import io
import traceback
from dotenv import load_dotenv
from stable_baselines3 import PPO
from sentence_transformers import SentenceTransformer
from retrieval_agent import RetrievalAgent
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient # Import the direct client

def main():
    load_dotenv()
    print("NexaCorp Agent System starting up...")

    try:
        comm_agent_model = PPO.load("communication_agent_model", device="cpu")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        DOMAINS = ["HR", "IT", "Payroll"]

        # --- NEW: Create a direct Hugging Face API client ---
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face API token not found in .env file")
            
        client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)
        print("✅ Hugging Face Inference Client created.")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="hkunlp/instructor-large",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Pass the direct client to the Retrieval Agent
        retrieval_agent = RetrievalAgent(client, embeddings)

    except Exception as e:
        print(f"An error occurred during initialization: {e}")
        traceback.print_exc()
        return

    print("\nNexaCorp Multi-Agent System is Online. Type 'exit' to quit.")

    while True:
        try:
            user_query = input("\nUser: ")
            if user_query.lower() == "exit":
                break
            if not user_query.strip():
                continue

            query_embedding = embedding_model.encode(user_query)
            action_index, _ = comm_agent_model.predict(query_embedding, deterministic=True)
            chosen_domain = DOMAINS[action_index]
            
            command = {"domain": chosen_domain, "query_text": user_query}
            print(f"CA -> RA | Command: {command}")

            answer = retrieval_agent.execute_task(command)
            print(f"\nAnswer: {answer}")

        except Exception as e:
            print(f"\nAn error occurred during processing.")
            traceback.print_exc()
            print("Please try rephrasing your question.")
        
        finally:
            print("\n" + "-"*70)

if __name__ == "__main__":
    main()