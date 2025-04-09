import os
import mlflow
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# --- Configuration ---
GROQ_API_KEY = "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"
MODEL_NAME = "qwen-2.5-32b"
EXPERIMENT_NAME = "langchain_groq_demo"

# --- Initialize MLflow ---
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.langchain.autolog()

# --- Initialize LLM ---
llm = ChatGroq(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY
)

# --- Define Prompt Template ---
prompt = PromptTemplate.from_template("Answer the following question: {question}")
chain = prompt | llm

# --- Run the Chain and Log to MLflow ---
with mlflow.start_run():
    question = "What is MLflow?"
    result = chain.invoke({"question": question})
    
    # Log to MLflow
    mlflow.log_param("model", MODEL_NAME)
    mlflow.log_param("question", question)
    mlflow.log_text(result.content, "response.txt")
    
    # Print response
    print("Response from LLM:\n", result.content)
