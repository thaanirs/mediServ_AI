from typing import Optional
import random
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from opengpt.config import Config
from opengpt.model_utils import add_tokens_to_model_and_tokenizer
import pymongo
import torch
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
from fuzzywuzzy import fuzz

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins='*',
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = pymongo.MongoClient("mongodb+srv://vedant11:vedant11@gathertube.zku14hn.mongodb.net/")
db = client["test"]
collection = db['user_data']
# df = {
#     'doctor_name': [],
#     'id': [],
#     'rating': [],
#     'distance': [],
#     'slots': [],
#     'department': []
# }
# for i in collection.find():
#         try:
#                 df["doctor_name"].append(i["ailments"][-1]["doctor"]["name"])
#                 df["id"].append(i["id"])
#                 df["rating"].append(i["ailments"][-1]["doctor"]["rating"])
#                 df["distance"].append(random.randrange(0,5))
#                 t =random.randint(1,12)
#                 slot = random.choice(['AM','PM'])
#                 df["slots"].append("{}{} - {}{}".format(t,slot,t+2,slot))
#                 df["department"].append(i["ailments"][-1]["doctor"]["type"])
#         except KeyError:
#                 continue
# for i in df.keys():
#     print(len(df[i]))

config = Config(yaml_path='./example_train_config.yaml')

tokenizer_bot = AutoTokenizer.from_pretrained('./tokenizer' )

tokenizer_bot.model_max_length = config.train.max_seq_len
model_bot = AutoModelForCausalLM.from_pretrained("./model")

from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("czearing/article-title-generator")
# model = AutoModel.from_pretrained("czearing/article-title-generator")
# pipe = pipeline("text2text-generation", model=model)


add_tokens_to_model_and_tokenizer(config, tokenizer_bot, model_bot)

gen = pipeline(model=model_bot, tokenizer=tokenizer_bot, task='text-generation', device=model_bot.device)


tokenizer_art = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model_art = BioGptForCausalLM.from_pretrained("microsoft/biogpt")


special_token = "#"

positive = ['The Advantages of Possessing good level ',
'Discover the Benefits of Having borderline ',
'Understanding the Upsides of Holding low level',
'Gaining from the Virtues of Owning ' 
'The Positive Side of low',
'Exploring the Pluses of good level of',
'Delving into the Profit of Possessing',
'Making the Most of Having',
'The Unexpected Bonuses of Retaining',
'Exploring the Benefits of Having',
'Why You Should Consider Getting',
'Appreciating the Assets of Owning',
'The Magic of Having',
'The Power and Potential of Possessing',
'Embrace the Gains of Securing',
'How Owning Can Enrich Your Life',
'The Hidden Advantages of Having']

negative=['Navigating the Complications of Living with Diabetes'
,'Unpacking the Struggles of Diabetic Life'
,'Understanding the Challenges of Diabetes Management'
,'The Hidden Troubles of Being Diabetic'
,'Coping with the Difficulties of Diabetes'
,'The Downside of Living with Diabetes'
,'Grappling with the Complexities of Diabetes'
,'Living in the Shadow of Diabetes: The Challenges and Struggles'
,'Behind the Diagnosis: The Realities of Living with Diabetes'
,'The Daily Struggles of Managing Diabetes' 
,'The Obstacles and Issues of Having Diabetes'
,'Encountering the Trials of Diabetes: A Comprehensive Overview'
,'The Tough Road: Dealing with Diabetes'
,'When Sugar Betrays: The Tribulations of Diabetes'
,'The Silent Battle: Facing the Challenges of Diabetes'
,'Sugar and Sorrow: The Dilemmas of Diabetes'
,'The Harsh Realities of Dealing with Diabetes'
,'The Unseen Battles: Confronting Diabetes Head-On'
,'Sugarcoated Struggles: The Hidden Challenges of Diabetes'
,'The Everyday Hurdles for People Living with Diabetes']

model_intro = {
    "Intro 1":
"Beep! Your health comes first. Stay informed."

,"Intro 2":
"Buzz, buzz! your personalized health assistants, always by your side."

,"Intro 3":
"Stay on top of your medical needs with the gentle reminders "

,"Intro 4":
"Be alerted to important health updates "

,"Intro 5":
"Your well-being matters. Let our medical phone notifications guide you towards a healthier tomorrow."

,"Intro 6":
"Experience the power of knowledge with our medical phone notifications, tailored to your health goals."

,"Intro 7":
"Stay connected to your healthcare journey."

,"Intro 8":
"From medication reminders to vital health insights"

,"Intro 10":
"Your health, our priority. Trust our medical phone notifications to keep you in the pink of health."
}

change_bh = {
    "positive":positive,  
    "negative":negative
}

chatHistory = []
currentQuery = ''

class bot_item(BaseModel):
    query : str
class not_item(BaseModel):
    # current : Optional[str]
    # old:Optional[str]
    # trait:Optional[str]
    min_len : int 
    max_len : int
    patient_id : int

class art_item(BaseModel):
    keyword : str
    min_len : Optional[int]
    max_len : Optional[int] 

class recommend_item(BaseModel):
    # input_symptoms:str 
    # input_disease:str 
    # input_doctor:str 
    # input_department:str 
    # input_severity:str
    # input_medical_test:str
    patient_id :int

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
        # response = response[response.rfind("<|ai|>")+len("<|ai|>"):]
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

# def contentGenerate(query,min_len,max_len):
#     try :
#         global tokenizer_art, model_art
#         inputs = tokenizer_art(query, return_tensors="pt")
#         set_seed(42)
#         with torch.no_grad():
#             beam_output = model_art.generate(**inputs,
#                                         min_length=min_len,
#                                         max_length=max_len,
#                                         num_beams=5,
#                                         early_stopping=True
#                                         )
#         response = tokenizer_art.decode(beam_output[0], skip_special_tokens=True)
#         return { "article":response }
#     except :
#         return {"article" : ""}
    
def compare(current,past):
    mapsevere = {
        "very high" :3,
        "high":2,
        "border":1,
        "medium":1,
        "low":0,
        "":-1
    }
    current = mapsevere[current]
    past = mapsevere[past]
    if current < past :
        return 1
    elif current > past :
        return -1
    return 0

# def Notify(old,new,trait):
#     headline = random.choice(list(model_intro.values()))
#     change = compare(old,new)
#     if change >0:
#         # choose random intro
#         userquery = random.choice((change_bh['positive'])) + "{}".format(trait)
#         # model.gen(userquery,)
#     elif change < 0:
#             userquery = random.choice(change_bh['negative']) + "{}".format(trait)
#     print(userquery)
#     try :
#         global tokenizer_art, model_art
#         inputs = tokenizer_art(userquery, return_tensors="pt")
#         set_seed(42)
#         with torch.no_grad():
#             beam_output = model_art.generate(**inputs,
#                                         min_length=50,
#                                         max_length=300,
#                                         num_beams=5,
#                                         early_stopping=True
#                                         )
#         response = tokenizer_art.decode(beam_output[0], skip_special_tokens=True)
#         return { "headline":headline,"article":response }
#     except :
#         return {"headline":headline,"article" : ""}

def getUserQuery(patient_id):
    print("notify called")
    patient_data=''
    print(patient_id)
    for i in collection.find({"id":patient_id}):
        patient_data = i
    if patient_data == '':
        print("patient no foun")
    if "ailments" in patient_data:
        if len(patient_data["ailments"]) > 1:
            old = patient_data["ailments"][-2]["severity"]
            new = patient_data["ailments"][-1]["severity"]
        else:
            new = patient_data["ailments"][-1]["severity"]
            old = patient_data["ailments"][-1]["severity"]
        trait = patient_data["ailments"][-1]['name']
        change = compare(old,new)
        print(change)
        age = 45
        ifgender = ''
        ifage=''
        if age < 30:
            ifage = 'young '
        elif age> 60 :
            ifage = 'old'
        else:
            ifage = 'middle aged'
        if change >=0:
            userquery = random.choice((change_bh['positive'])) + " {} ".format(trait) + "for  {} {}".format(ifage,ifgender) 
        elif change < 0:
                userquery = random.choice(change_bh['negative']) + " {} ".format(trait)
        return (userquery)
    

def Notify(patient_id,min_len=100,max_len=500):
    headline = random.choice(list(model_intro.values()))
    try:
        userquery = getUserQuery(patient_id)
    except:
        userquery = ""
    try :
        global tokenizer_art, model_art
        inputs = tokenizer_art(userquery, return_tensors="pt")
        set_seed(42)
        with torch.no_grad():
            beam_output = model_art.generate(**inputs,
                                        min_length=min_len,
                                        max_length=max_len,
                                        num_beams=5,
                                        early_stopping=True
                                        )
        response = tokenizer_art.decode(beam_output[0], skip_special_tokens=True)
        return { "headline":headline,"article":response }
    except :
        return {"headline":headline,"article" : ""}

def find_best_match(Id,input_symptoms, input_disease, input_doctor, input_department, input_severity, input_medical_test):
    print("called here as well")
    patients_data = collection.find()
    input_data = {
        "id":Id,
        'symptoms': input_symptoms,
        'disease_name': input_disease,
        'doctor_name': input_doctor,
        'department': input_department,
        'severity': input_severity,
        'medical_tests': input_medical_test
        # 'medical_tests': ''
    }

    if not isinstance(input_data['symptoms'], list):
        input_data['symptoms'] = [input_data['symptoms']]
    if not isinstance(input_data['medical_tests'], list):
        input_data['medical_tests'] = [input_data['medical_tests']]

    similarity_percentages = []
    for patient_data in patients_data:
        print(patient_data)
        if patient_data["id"] != input_data["id"] and patient_data["id"] not in [0,4,5,8]:
            patient_scores = []
            # for past_disease in [patient_data]:
            a = ','.join(input_data['symptoms'])
            try:
                b = ','.join(patient_data["ailments"][-1]['symptoms'])
            except  :
                continue
            # except TypeError:
            #     continue
            symptom_score = fuzz.token_set_ratio(a, b)
            disease_score = fuzz.token_sort_ratio(input_data['disease_name'], patient_data['ailments'][-1]["name"])
            doctor_score = fuzz.token_sort_ratio(input_data['doctor_name'], patient_data["ailments"][-1]['doctor']["name"])
            department_score = fuzz.token_sort_ratio(input_data['department'], patient_data["ailments"][-1]['doctor']["type"])
            severity_score = fuzz.token_sort_ratio(input_data['severity'], patient_data['ailments'][-1]["severity"])
            c=','.join(input_data['medical_tests'])
            # for i in range(len(patient_data["ailments"][-1]['lab_test'])):
            #     if patient_data["ailments"][-1]['lab_test'][i] == None:
            #         patient_data["ailments"][-1]['lab_test'][i]=""
            if None in patient_data["ailments"][-1]['lab_test']:
                continue
            d =','.join(patient_data["ailments"][-1]['lab_test'])
            medical_tests_score = fuzz.token_set_ratio(c, d)
            total_score = (symptom_score + disease_score + doctor_score + department_score + severity_score + medical_tests_score) / 6
            patient_scores.append(total_score)
            similarity_percentages.append(max(patient_scores))

    # print("asdasdad")
    print(similarity_percentages)
    # return similarity_percentages
    max_sim =max(similarity_percentages) 
    print(max_sim)
    if max_sim< 70:
      return "No Match"
    return similarity_percentages
    

def Recommend(patient_id):
    # try:
        print("called here")
        patient_data = dict()
        for i in collection.find({"id":patient_id}):
            print("i is",i)
            patient_data = i
        if "ailments" not in patient_data:
            return {"response","Try a different user"}
        input_symptoms = patient_data["ailments"][-1]["symptoms"]
        input_disease = patient_data["ailments"][-1]["name"]
        input_doctor = patient_data["ailments"][-1]["doctor"]["name"]
        # input_department = patient_data["ailments"][-1]["doctor"]["name"]
        input_department = patient_data["ailments"][-1]["doctor"]["type"]
        input_severity = patient_data["ailments"][-1]["severity"]
        input_medical_test = patient_data["ailments"][-1]["lab_test"]
        
        best_match_patient = find_best_match(patient_id,input_symptoms, input_disease, input_doctor, input_department, input_severity, input_medical_test)
        # return {"match score":best_match_patient,"patient_data":patient_data}
        # return {"patient_data":patient_data}
        if best_match_patient == "No Match":
            return "No match"
        patients_data = collection.find()
        recommendedpat = patients_data[best_match_patient.index(max(best_match_patient))]
        print("match",type(recommendedpat),recommendedpat)
        print({"match score":best_match_patient,"patient_data":patient_data})
        # return {"match score":best_match_patient,"patient_data":patient_data}
        # return recommendedpat
        print("Best Match:")
        return (recommendedpat)
    # except KeyError:
    #     print("Insufficient data")

def find_recommended_doctor(patient_id, symptoms, disease, emergency):
    # global df
    df = pd.read_csv("./data.csv")
    print(type(df))
    df = pd.DataFrame(df)
    matching_doctors = df[
        (df['department'] == disease) |
        (df['department'].apply(lambda x: fuzz.token_set_ratio(disease, x)) > 70)
    ]
    # print(df.dep)
    if emergency:
        recommended_doctor = matching_doctors.sort_values(by='distance').iloc[0]
    else:
        matching_doctors['symptom_score'] = matching_doctors['doctor_name'].apply(lambda x: fuzz.token_set_ratio(','.join(symptoms), x))
        recommended_doctor = matching_doctors.sort_values(by=['symptom_score', 'rating'], ascending=[False, False]).iloc[0]

    return recommended_doctor

def recRec(patient_id):
    patient_id = 1
    patient_data = ''
    for i in collection.find({"id":patient_id}):
        patient_data = i
    if "ailments" in patient_data:
        symptoms = patient_data["ailments"][-1]['symptoms']
        disease = patient_data["ailments"][-1]['name']
        emergency = False

        recommended_doctor = find_recommended_doctor(patient_id, symptoms, disease, emergency)
        print("Recommended Doctor:")
        return (recommended_doctor)
    else:
        return "could not find"

@app.get("/")
def read_root():
    return {"Hello": "World"}



@app.post("/medibot/")
async def botResponse(item:bot_item):
    # print(item)
    return mainBot(item.query)

# @app.post("/article/")
# async def content(item:art_item):
#     # print(item)
#     return contentGenerate(item.keyword,item.min_len,item.max_len)

@app.post("/rel/")
async def Rel(query1:str,query2:str):
    return find_relevance(query1,query2)

@app.post("/medibot_2.0/")
async def botsResponse(item:bot_item):
    return mainBot_2(item.query)

@app.post("/notification/")
async def notify(item:not_item):
    # return Notify(item.old,item.current,item.trait)
    print("notify called",item)
    return Notify(item.patient_id,item.min_len,item.max_len)  

@app.post("/recommend/")
async def recommend(item:recommend_item):
    print("called")
    return Recommend(item.patient_id)

@app.post("/rec/")
async def Rec(item:recommend_item):
    return recRec(item.patient_id)