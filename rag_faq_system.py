import time
import logging
from typing import List, Dict, Tuple
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
import tiktoken
import argparse
import os

# --- Configuration ---
DEFAULT_TOP_K = 6
MAX_CONTEXT_TOKENS = 1800   # For GPT-3.5-turbo (4096 context), leave ample room for instructions and answer
LLM_CONTEXT_LIMIT = 4096
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_LLM_MODEL = "gpt-3.5-turbo"
CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_faq_db")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger("FAQ_RAG")

# --- Token Counting Utility ---
def count_tokens(text: str, model: str = LLM_CONTEXT_LIMIT) -> int:
    # tiktoken can use encoding_for_model
    encoding = tiktoken.encoding_for_model(OPENAI_LLM_MODEL)
    return len(encoding.encode(text))

# --- Query Encoder ---
def encode_query(query: str) -> List[float]:
    """Encodes the query using OpenAI embeddings (API)."""
    response = openai.Embedding.create(
        input=query,
        model=OPENAI_EMBEDDING_MODEL
    )
    return response['data'][0]['embedding']

# --- Retriever ---
def retrieve_docs(
    query: str,
    vectorstore: Chroma,
    top_k: int = DEFAULT_TOP_K
) -> Tuple[List[Document], float]:
    start = time.time()
    # Use similarity_search_with_score for access to scores
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=top_k)
    latency = time.time() - start
    docs = [doc for doc, score in docs_and_scores]
    logger.info(f"Retriever latency: {latency:.3f}s | Num docs: {len(docs)}")
    return docs, latency

# --- Context Builder: Token-budgeted, with Citation Markers ---
def build_context(
    docs: List[Document],
    token_budget: int = MAX_CONTEXT_TOKENS,
    llm_model: str = OPENAI_LLM_MODEL
) -> Tuple[str, List[int]]:
    """
    Assembles as much context as possible under token budget, each chunk gets [#] for citation.
    Returns the context string and included doc indices for later citation.
    """
    encoding = tiktoken.encoding_for_model(llm_model)
    assembled = []
    total_tokens = 0
    used_indices = []
    for idx, doc in enumerate(docs, start=1):
        context_string = doc.page_content.strip()
        # Add citation marker
        context_with_citation = f"[{idx}] {context_string}"
        tokens = len(encoding.encode(context_with_citation))
        if total_tokens + tokens > token_budget:
            break
        assembled.append(context_with_citation)
        used_indices.append(idx)
        total_tokens += tokens
    logger.info(f"Assembled context with {len(assembled)} chunks, total {total_tokens} tokens.")
    return '\n\n'.join(assembled), used_indices

# --- Prompt Formatter ---
def format_prompt(
    question: str, 
    context: str,
    citation_indices: List[int]
) -> str:
    prompt = (
        "You are an expert enterprise FAQ assistant. "
        "Answer the question below using ONLY the context provided. "
        "Cite the sources in brackets, e.g. [1], in your answer wherever facts from context are used. "
        "Do not make up information not present in the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (include [n] markers to indicate source, be concise and accurate):"
    )
    return prompt

# --- LLM Call ---
def call_llm(prompt: str, llm_model: str = OPENAI_LLM_MODEL) -> Tuple[str, int, float]:
    openai.api_key = os.environ['OPENAI_API_KEY']
    start = time.time()
    response = openai.ChatCompletion.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.2,
    )
    latency = time.time() - start
    text = response['choices'][0]['message']['content']
    usage = response['usage']
    total_tokens = usage['total_tokens']
    logger.info(f"LLM latency: {latency:.2f}s | Token usage: {total_tokens}")
    return text, total_tokens, latency

# --- Pipeline ---
def answer_query(
    query: str,
    vectorstore: Chroma,
    top_k: int = DEFAULT_TOP_K,
    token_budget: int = MAX_CONTEXT_TOKENS,
    llm_model: str = OPENAI_LLM_MODEL
) -> Dict:
    # 1. Retrieve
    docs, retrieval_latency = retrieve_docs(query, vectorstore, top_k=top_k)
    hit_rate = len(docs) / top_k if top_k else 0
    # 2. Context Assembly
    context, citation_indices = build_context(docs, token_budget=token_budget, llm_model=llm_model)
    # 3. Prompt
    prompt = format_prompt(query, context, citation_indices)
    token_count = count_tokens(prompt)
    # 4. LLM Call
    llm_answer, used_tokens, llm_latency = call_llm(prompt, llm_model=llm_model)

    result = {
        "query": query,
        "retrieved": [d.page_content for d in docs],
        "context": context,
        "answer": llm_answer,
        "citations": citation_indices,
        "retriever_latency_s": retrieval_latency,
        "llm_latency_s": llm_latency,
        "retrieval_hit_rate": hit_rate,
        "prompt_tokens": token_count,
        "used_tokens": used_tokens,
    }
    return result

# --- Main CLI interface ---
def main():
    parser = argparse.ArgumentParser(description="Enterprise FAQ RAG System (Chroma/LLM)")
    parser.add_argument('--question', type=str, required=False, help="FAQ question to answer")
    parser.add_argument('--top_k', type=int, default=DEFAULT_TOP_K)
    parser.add_argument('--token_budget', type=int, default=MAX_CONTEXT_TOKENS)
    args = parser.parse_args()

    openai.api_key = os.environ['OPENAI_API_KEY']

    logger.info(f"Connecting to Chroma vectorstore at {CHROMA_PERSIST_DIRECTORY}...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY']),
    )
    logger.info("Chroma vectorstore loaded.")

    # Simple REPL if question not supplied
    if not args.question:
        print("Type your FAQ question (Ctrl+C to exit):")
        while True:
            try:
                question = input("\nQ: ")
                result = answer_query(
                    question, vectorstore,
                    top_k=args.top_k,
                    token_budget=args.token_budget
                )
                print(f"\nAI Answer (with citations): {result['answer']}\n")
                print(f"[Diagnostics] Retrieval hit rate: {result['retrieval_hit_rate']:.2f}, Retriever: {result['retriever_latency_s']:.2f}s, LLM: {result['llm_latency_s']:.2f}s, Prompt tokens: {result['prompt_tokens']}, Used tokens: {result['used_tokens']}")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    else:
        result = answer_query(
            args.question, vectorstore,
            top_k=args.top_k, token_budget=args.token_budget
        )
        print(f"AI Answer (with citations): {result['answer']}\n")
        print(f"[Diagnostics] Retrieval hit rate: {result['retrieval_hit_rate']:.2f}, Retriever: {result['retriever_latency_s']:.2f}s, LLM: {result['llm_latency_s']:.2f}s, Prompt tokens: {result['prompt_tokens']}, Used tokens: {result['used_tokens']}")
        

# --- Entry Point ---
if __name__ == "__main__":
    main()
