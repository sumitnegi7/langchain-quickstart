import {BufferMemory} from "langchain/memory"
import {ChatOpenAI} from "langchain/chat_models/openai"
import {ConversationChain } from "langchain/chains"
import {RedisChatMessageHistory} from "langchain/stores/message/redis"
import { config } from "dotenv";
config()
// Memory

const memory = new BufferMemory({
    chatHistory: new RedisChatMessageHistory({
        sessionId: '123',
        sessionTTL: 600,
        config: {
            password: process.env.REDIS_PASSWORD,
            socket: {
                host: process.env.REDIS_HOST,
                port: 16031
            }
        }
    })
});
import { config } from "dotenv";
config();

// Model
const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPEN_API_KEY,
    temperature: 0
});

// Chain
const chain = new ConversationChain({
    llm: model,
    memory: memory
})

const res = await chain.call({
    input: "What is my name"
})

console.log(res)