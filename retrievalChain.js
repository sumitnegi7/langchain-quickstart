import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";

// Document loading using cheerio
const loader = new CheerioWebBaseLoader(
  "https://docs.smith.langchain.com/overview"
);

const docs = await loader.load();
// splitiing docs to feed to vectorstore
const splitter = new RecursiveCharacterTextSplitter();
const splitDocs = await splitter.splitDocuments(docs);


/// converts Objects//string to vectors
const embeddings = new OpenAIEmbeddings({
    openAIApiKey:  process.env.OPEN_API_KEY,
});
const chatModel = new ChatOpenAI({
    openAIApiKey: process.env.OPEN_API_KEY,
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

// Setup retrieval chain to fetch the docs
/**
 * Create a retrieval chain that retrieves documents and then passes them on.

@param params
A params object containing a retriever and a combineDocsChain.

@returns
An LCEL Runnable which returns a an object containing at least context and answer keys.

@example
 */
const retrievalChain = await createRetrievalChain({
  combineDocsChain: documentChain,
  retriever,
});

const result = await retrievalChain.invoke({
    input: "what is LangSmith?",
  });

  const result2 = await retrievalChain.invoke({
    input: "can you repeat what you said ?",
  });
  
  console.log("💎 💎 💎 ",result2.answer,"💎 💎 💎 "); 