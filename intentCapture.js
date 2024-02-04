import { DynamicTool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { config } from "dotenv";
config();
import { AgentExecutor, createOpenAIToolsAgent } from "langchain/agents";
import { pull } from "langchain/hub";
import { RunnableSequence } from "@langchain/core/runnables";
import { formatToOpenAIFunctionMessages } from "langchain/agents/format_scratchpad";
import { OpenAIFunctionsAgentOutputParser } from "langchain/agents/openai/output_parser";
import {
    ChatPromptTemplate,
    MessagesPlaceholder,
  } from "@langchain/core/prompts";
import { convertToOpenAIFunction } from "@langchain/core/utils/function_calling";

const model = new ChatOpenAI({
    openAIApiKey: process.env.OPEN_API_KEY,
  });
  
const intentAnalysisTool = new DynamicTool({
 name: "intent_analysis",
 description: "Analyzes user input to determine intent and if user intents to buy asks them for contact number.",
 func: async (input) => {
    console.log("🚀 ~ func: ~ input:", input)
    // Analyze the input and determine the intent
    const intent = analyzeInput(input);
    console.log("🚀 ~ func: ~ intent:", intent)

    if(intent === "xyz"){
       // call An external function
    }
    return intent;
 },
});



const tools = [intentAnalysisTool];
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "Answer user queries and if user intends on buying ask him for phone number"],
    ["human", "{input}"],
    new MessagesPlaceholder("agent_scratchpad"),
  ]);


  const modelWithFunctions = model.bind({
    functions: tools.map((tool) => convertToOpenAIFunction(tool)),
  });


const runnableAgent = RunnableSequence.from([
  {
    input: (i) => i.input,
    agent_scratchpad: (i) =>
      formatToOpenAIFunctionMessages(i.steps),
  },
  prompt,
  modelWithFunctions,
  new OpenAIFunctionsAgentOutputParser(),
]);

const executor = AgentExecutor.fromAgentAndTools({
  agent: runnableAgent,
  tools,
});



const input = "I wanna buy ps5";
console.log(`Calling agent executor with query: ${input}`);

const result = await executor.invoke({
  input,
});

console.log(result);