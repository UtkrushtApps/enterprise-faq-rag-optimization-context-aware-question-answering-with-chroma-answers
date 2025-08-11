# Solution Steps

1. Install required dependencies: langchain, chromadb, openai, tiktoken, argparse, etc.

2. Set your OpenAI API key in the environment variable OPENAI_API_KEY. Also set CHROMA_PERSIST_DIRECTORY as needed to connect to your pre-populated Chroma vectorstore.

3. Create a new Python script, e.g., 'rag_faq_system.py'.

4. Implement robust logging for retrieval and generation steps, tracking retrieval latency, LLM latency, token usage, and retrieval hit rate.

5. Set up the query encoder using OpenAI's embedding API, matching the embedding approach used for your FAQ documents.

6. Initialize the Chroma vectorstore using langchain's Chroma class, loading from the configured persistence directory, and attach the OpenAIEmbeddings function.

7. Define a retriever function that uses similarity_search_with_score to perform a top-k dense similarity search (cosine or dot-product, per Chroma config), returning the closest FAQ chunks.

8. Implement a robust context builder: for the retrieved FAQ docs, assemble as many as possible without exceeding a token budget (e.g., 1800 tokens), add citation markers ([1], [2], etc.), and concatenate context chunks.

9. Count tokens in the context and prompt using tiktoken, adapt context assembly to respect available context size for the model.

10. Write a prompt formatting function instructing the LLM to only use information present in the cited context and to use [n] citation markers in the answer.

11. Wire the retrieval, context assembly, prompt formatting, and LLM call into an answer_query() pipeline; this function should collect and log all relevant metrics for diagnostics.

12. Implement a command-line interface (CLI) that takes a question as an argument or supports an interactive mode.

13. In this CLI, on each query: run the pipeline, display the answer with inline citations, and print diagnostic information (retrieval hit rate, retrieval latency, LLM latency, token usage).

14. Test the pipeline with various queries to verify accuracy, context window handling, and citation formatting.

