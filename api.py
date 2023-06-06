import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
from fastapi.responses import JSONResponse

app = FastAPI()

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
data = load_model()
regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


@app.get('/')
def index():
    return {'message':'Hello,Amaan'}

@app.get('/GetSalary')
def get_salary(country:str,education:str,experience:int):
    X = np.array([[country, education, experience ]])
    X[:, 0] = le_country.transform(X[:,0])
    X[:, 1] = le_education.transform(X[:,1])
    X = X.astype(float)
    salary = regressor_loaded.predict(X)
    return salary[0]

# run the api with uvicorn
if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)
