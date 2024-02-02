import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";

const chatModel = new ChatOpenAI({
    openAIApiKey: process.env.OPEN_API_KEY,
  });

  
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