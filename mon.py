import pymongo
client = pymongo.MongoClient("mongodb+srv://vedant11:vedant11@gathertube.zku14hn.mongodb.net/")
db = client["test"]
collection = db['user_data']
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
# print(list(collection.find({},{"patient_id":100})))
print(client.list_database_names())
print(db.list_collection_names())

data = ''
for i in (collection.find({'id':10})):
    print(i)
    data = i
print(type(data))