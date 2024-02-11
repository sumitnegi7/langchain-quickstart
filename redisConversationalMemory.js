import { BufferMemory } from "langchain/memory";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { ConversationChain } from "langchain/chains";
import { RedisChatMessageHistory } from "langchain/stores/message/redis";
import { config } from "dotenv";
import cors from "cors";
import { CallbackManager } from "langchain/callbacks";

config();

import express from "express";

const app = express();
const port =  5000;
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

app.get('/api/chat', async (req, res) => {
  // Check if the query parameter exists
  if (!req.query.qs) {
    res.status(400).send('Missing query parameter qs');
    return;
  }

  // Set the necessary headers for SSE
  const headers = {
    "Content-Type": "text/event-stream",
    Connection: "keep-alive",
    "Cache-Control": "no-cache",
  };
  res.writeHead(200, headers);

  const memory = new BufferMemory({
    chatHistory: new RedisChatMessageHistory({
      sessionId: '123',
      sessionTTL:  600,
      config: {
        password: process.env.REDIS_PASSWORD,
        socket: {
          host: process.env.REDIS_HOST,
          port:  16031
        }
      }
    })
  });

  const model = new ChatOpenAI({
    maxTokens:  300,
    streaming: true,
    modelName: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPEN_API_KEY,
    temperature:  0,
    callbackManager: CallbackManager.fromHandlers({
      async handleLLMNewToken(token) {
        // Write each token as an SSE message
        res.write(`data: ${JSON.stringify({ type: "stream", sender: "bot", chunk: token })}\n\n`);
      },
      async handleLLMEnd(output) {
        // End the stream after all tokens have been sent
        res.end();
        console.log("End of stream.", output);
      },
    }),
  });

  const chain = new ConversationChain({
    llm: model,
    memory: memory
  });

  await chain.call({
    input: req.query.qs
  });

  // No need to log the result here since it's already logged in the callbacks
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
