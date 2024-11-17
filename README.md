Here's a **README.md** template for your chatbot project:

---

# Context-Aware Chatbot with Retrieval-Augmented Generation (RAG)

This project implements a **Context-Aware Chatbot** using the **Retrieval-Augmented Generation (RAG)** technique, enabling the bot to answer user queries based on a given text corpus. The system remembers context from previous interactions and dynamically rephrases queries to optimize the answer generation.

The chatbot uses a pipeline for **question rephrasing** to ensure that it can adapt to the context and provide accurate responses even if the query is slightly altered.

## Technologies Used

- **Python**: The primary programming language used to build the chatbot system.
- **Poetry**: For managing project dependencies and packaging.
- **Langchain**: For orchestrating the logic and managing the RAG pipeline.
- **Gemini API**: Utilized for querying and interacting with external AI models for response generation.
- **Sentence Encoder**: For creating embeddings that allow the chatbot to effectively understand and retrieve relevant information from the corpus.

## Features

- **Context Awareness**: The chatbot can remember previous interactions, ensuring it maintains context throughout a conversation.
- **Query Rephrasing**: The system rephrases incoming queries based on the context to enhance the quality of responses.
- **RAG Integration**: Combines retrieval and generation methods to answer queries based on provided text data.
- **Dynamic Response Generation**: Uses state-of-the-art APIs and models to generate responses dynamically based on user input.

## Usage

- **Ask a Question**: Once the chatbot is running, you can ask it questions based on the given text data. The bot will use its contextual memory to provide relevant answers.
- **Contextual Interaction**: The chatbot will remember context from previous queries in the session, making the conversation more natural and coherent.
- **Rephrasing Queries**: The system intelligently rephrases queries before passing them through the RAG pipeline to ensure optimal responses.

## Example Workflow

1. **Initial Query**:
   User: "What is the capital of France?"

2. **Response**:
   Bot: "The capital of France is Paris."

3. **Contextual Follow-up**:
   User: "And how far is it from Berlin?"

4. **Response**:
   Bot: "Paris is approximately 1,050 km away from Berlin."
