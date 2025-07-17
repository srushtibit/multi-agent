
# Language-Emergent Multi-Agent AI System for Knowledge-Based Issue Resolution

This repository contains the source code for the Major Project: "Language-Emergent Multi-Agent AI System for Knowledge-Based Issue Resolution Using RAG over Enterprise Documents." The project demonstrates a novel two-agent architecture where a **Communication Agent** learns to delegate tasks to a **Retrieval Agent** to answer user queries based on a private knowledge base.

##  Key Features

  * **Multi-Agent Architecture:** A robust system composed of a user-facing Communication Agent and a backend Retrieval Agent.
  * **Reinforcement Learning-based Routing:** The Communication Agent is trained using Reinforcement Learning (PPO algorithm) to analyze user queries and intelligently route them to the correct specialist.
  * **Specialized RAG Agents:** The Retrieval Agent uses multiple Retrieval-Augmented Generation (RAG) chains, each an expert on a specific enterprise document (e.g., HR, IT, Payroll manuals).
  * **Emergent Communication Protocol:** The agents communicate via a simple, efficient, and learned protocol, where the Communication Agent sends structured commands instead of relying on complex natural language.

-----
##  System Architecture

The system operates based on a two-agent design:

1.  **Communication Agent (CA):** The "brain" of the operation. It receives the user's query and, using its RL training, decides which specialist is best suited to handle the task.
2.  **Retrieval Agent (RA):** The "engine" of the operation. It receives a direct command from the CA and uses the appropriate specialized RAG pipeline to find and generate an answer from the knowledge base.

This clear separation of duties allows for efficient and accurate issue resolution.

-----

##  Technology Stack

  * **Primary Language:** Python
  * **Reinforcement Learning:** Stable-Baselines3 (with PyTorch)
  * **RL Environment:** Gymnasium
  * **LLM & Embeddings Framework:** LangChain, LangChain-Huggingface
  * **NLP & Embeddings:** Sentence-Transformers
  * **Vector Store:** FAISS (Facebook AI Similarity Search)
  * **LLM & Model Hosting:** Hugging Face Hub
  * **Document Loading:** python-docx, pandas

-----

##  Setup and Installation

Follow these steps to set up and run the project locally.

### 1\. Clone the Repository

```bash
git clone <your-repository-url>
cd major_project
```

### 2\. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3\. Install Dependencies

Install all the required Python libraries using the following command:

```bash
pip install gymnasium sb3-contrib torch sentence-transformers pandas python-docx faiss-cpu langchain langchain-huggingface python-dotenv docx2txt
```

### 4\. Set Up Environment Variables (`.env` file)

For security, we need to store our secret API key in a separate file that is not committed to version control.

1.  Create a file named `.env` in the root of the project directory.
2.  Open the file and add your Hugging Face API token in the following format:
    ```
    HUGGINGFACEHUB_API_TOKEN="hf_YourTokenHere"
    ```
3.  The `python-dotenv` library, which we installed, allows our Python scripts to automatically load this key when we call the `load_dotenv()` function. This keeps our secrets safe and out of the main codebase.

### 5\. Prepare the Dataset

1.  Create a folder named `dataset` in the project root.
2.  Place your enterprise documents (`.docx` files) inside this folder.
3.  Create a `training_data.csv` file inside the `dataset` folder with two columns: `query` and `correct_department` (`HR`, `IT`, or `Payroll`). This file is used to train the Communication Agent.

-----

##  How to Run

The project runs in two stages: training the agent and then running the main application.

### 1\. Train the Communication Agent

First, run the training script from your terminal. This will read your `training_data.csv` and create the `communication_agent_model.zip` file.

```bash
python train_communication_agent.py
```

### 2\. Run the Main Application

Once the agent is trained, you can start the interactive chatbot.

```bash
python main.py
```

The system will load the trained model and the knowledge base, and you can begin asking questions.