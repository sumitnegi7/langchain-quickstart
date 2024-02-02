import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { config } from "dotenv";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
// import { ChatOpenAI } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
// import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
// import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";
config();
// Document loading using cheerio
const loader = new CheerioWebBaseLoader(
  "https://docs.smith.langchain.com/overview"
);

const docs = await loader.load();
// splitiing docs to feed to vectorstore
const splitter = new RecursiveCharacterTextSplitter();
const splitDocs = await splitter.splitDocuments(docs);
const chatModel = new ChatOpenAI({
  openAIApiKey: process.env.OPEN_API_KEY,
});

/// converts Objects//string to vectors
const embeddings = new OpenAIEmbeddings({
    openAIApiKey:  process.env.OPEN_API_KEY,
});


  /// In memory vector store
const vectorstore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);



// indexed data in vector store => retrienval chain create (qs input)=>pass doc to llm


const prompt =
  ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}`);





/**
 * Create a chain that passes a list of documents to a model.

@param llm — Language model to use for responding.

@param prompt
 */
const documentChain = await createStuffDocumentsChain({
  llm: chatModel,
  prompt,
});


//Now that we have this data indexed in a vectorstore, we will create a retrieval chain. 
// This chain will take an incoming question, look up relevant documents, 
// then pass those documents along with the original question into an LLM and ask it to answer the original question.


const retriever = vectorstore.asRetriever();
config();


  
const historyAwarePrompt = ChatPromptTemplate.fromMessages([
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
  [
    "user",
    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
  ],
]);

const historyAwareRetrieverChain = await createHistoryAwareRetriever({
  llm: chatModel,
  retriever,
  rephrasePrompt: historyAwarePrompt,
});

import { HumanMessage, AIMessage } from "@langchain/core/messages";

const chatHistory = [
  new HumanMessage("Can LangSmith help test my LLM applications?"),
  new AIMessage("Yes!"),
];

const msg= await historyAwareRetrieverChain.invoke({
  chat_history: chatHistory,
  input: "Tell me how!",
});

console.log("🐃 🐃 🐃 🐃 msg",msg) 

const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "Answer the user's questions based on the below context:\n\n{context}",
  ],
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
]);

const historyAwareCombineDocsChain = await createStuffDocumentsChain({
  llm: chatModel,
  prompt: historyAwareRetrievalPrompt,
});

const conversationalRetrievalChain = await createRetrievalChain({
  retriever: historyAwareRetrieverChain,
  combineDocsChain: historyAwareCombineDocsChain,
});

const result2 = await conversationalRetrievalChain.invoke({
  chat_history: [
    new HumanMessage("Can LangSmith help test my LLM applications?"),
    new AIMessage("Yes!"),
  ],
  input: "tell me how",
});

console.log(result2.answer);