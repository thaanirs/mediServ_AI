from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from opengpt.config import Config
from opengpt.model_utils import add_tokens_to_model_and_tokenizer


import torch
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed

app = FastAPI()

config = Config(yaml_path='./example_train_config.yaml')

tokenizer_bot = AutoTokenizer.from_pretrained('./tokenizer' )

tokenizer_bot.model_max_length = config.train.max_seq_len
model_bot = AutoModelForCausalLM.from_pretrained("./model")

add_tokens_to_model_and_tokenizer(config, tokenizer_bot, model_bot)

gen = pipeline(model=model_bot, tokenizer=tokenizer_bot, task='text-generation', device=model_bot.device)


tokenizer_art = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model_art = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

print()

special_token = "==+=="

chatHistory = []
currentQuery = ''

class bot_item(BaseModel):
    query : str

class art_item(BaseModel):
    keyword : str
    min_len : Optional[int] = 150
    max_len : Optional[int] = 1024

def find_relevance(old,new):
    currentQuery = "<|user|>are these two issues connected or related? {} and  {} . If yes then provide a number between 0 to 1 representing percentage of relevance of the two issues<|eos|><|ai|>".format(old,new)
    response = gen(currentQuery, do_sample=True, max_length=250, temperature=0.2)[0]['generated_text']
    return response

def find_relevance2(old,new):
    currentQuery = "<|user|>are these two issues connected or related? {} and  {} . say yes or no<|eos|><|ai|>".format(old,new)
    response = gen(currentQuery, do_sample=True, max_length=250, temperature=0.2)[0]['generated_text']
    return {"response":response}

def mainBot(query):
    try:
        global currentQuery,special_token,chatHistory
        userInput = "<|user|>{}<|eos|><|ai|>".format(query)
        if chatHistory:
            currentQuery = chatHistory[-1]
        print("current query bfr is",currentQuery)
        if query.find(special_token)!=-1:
            currentQuery=''
            chatHistory=[]
            query = query.replace(special_token,"")
        currentQuery += userInput
        print("current query is",currentQuery)
        response = gen(currentQuery, do_sample=True, max_length=250, temperature=0.2)[0]['generated_text']
        print("response is",response)
        chatHistory.append(response[response.rfind("<|user|>"):])
        print(chatHistory)
        response = response[response.rfind("<|ai|>")+len("<|ai|>"):]
        return {"response":response}
    except :
        return {"response":'Sorry i could not understand you'}

def mainBot_2(query):
    try:
        global currentQuery,special_token,chatHistory
        userInput = "<|user|>{}<|eos|><|ai|>".format(query)
        if chatHistory:
            currentQuery = chatHistory[-1]
        print("current query bfr is",currentQuery)
        # if query.find(special_token)!=-1:
        #     currentQuery=''
        #     chatHistory=[]
        #     query = query.replace(special_token,"")
        print(find_relevance2(query,currentQuery))
        if find_relevance2(query,currentQuery) == "No":
            currentQuery=''
            chatHistory=[]
        currentQuery += userInput
        print("current query is",currentQuery)
        response = gen(currentQuery, do_sample=True, max_length=250, temperature=0.2)[0]['generated_text']
        print("response is",response)
        chatHistory.append(response[response.rfind("<|user|>"):])
        print(chatHistory)
        response = response[response.rfind("<|ai|>")+len("<|ai|>"):]
        return {"response":response}
    except :
        return {"response":'Sorry i could not understand you'}

def contentGenerate(query,min_len,max_len):
    try :
        global tokenizer_art, model_art
        inputs = tokenizer_art(query, return_tensors="pt")
        set_seed(42)
        with torch.no_grad():
            beam_output = model_art.generate(**inputs,
                                        min_length=min_len,
                                        max_length=max_len,
                                        num_beams=5,
                                        early_stopping=True
                                        )
        response = tokenizer_art.decode(beam_output[0], skip_special_tokens=True)
        return { "article":response }
    except :
        return {"article" : ""}


@app.get("/")
def read_root():
    return {"Hello": "World"}



@app.post("/medibot/")
async def botResponse(item:bot_item):
    # print(item)
    return mainBot(item.query)

@app.post("/article/")
async def content(item:art_item):
    # print(item)
    return contentGenerate(item.keyword,item.min_len,item.max_len)

@app.post("/rel/")
async def Rel(query1:str,query2:str):
    return find_relevance(query1,query2)

@app.post("/medibot_2.0")
async def botsResponse(item:bot_item):
    return mainBot_2(item.query)

