import { config } from "dotenv";
config();

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const chatModel = new ChatOpenAI({
  openAIApiKey: process.env.OPEN_API_KEY,
});

const res =await chatModel.invoke("what is LangSmith?");
console.log("🚀 ~ res:", res)

//  Prompt templates are used to convert raw user input to a better input to the LLM.


const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a world class technical documentation writer."],
  ["user", "{input}"],
]);

const chain = prompt.pipe(chatModel); // combining these using a simple LLM chain

const res2= await chain.invoke({
    input: "what is LangSmith?",
  });

  console.log("🚀🚀🚀 ~ res2:", res2,"🚀🚀🚀")

//   // Above output is that of a chain so using String Parser below

import { StringOutputParser } from "@langchain/core/output_parsers";

const outputParser = new StringOutputParser();

// To make it as easy as possible to create custom chains, we’ve implemented a “Runnable” protocol.
// The Runnable protocol is implemented for most components. 
// This is a standard interface, which makes it easy to define custom chains as well as invoke them in a standard way. 
// The standard interface includes:
// stream: stream back chunks of the response
// invoke: call the chain on an input
// batch: call the chain on a list of inputs


// /**
//      * Create a new runnable sequence that runs each individual runnable in series,
//      * piping the output of one runnable into another runnable or runnable-like.
//      * @param coerceable A runnable, function, or object whose values are functions or runnables.
//      * @returns A new runnable sequence.
//      */

const llmChain = prompt.pipe(chatModel).pipe(outputParser);

const res3= await llmChain.invoke({
  input: "what is LangSmith?",
});

console.log("res3🔥 🔥 🔥 ",res3) 