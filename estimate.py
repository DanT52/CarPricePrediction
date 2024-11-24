import cloudpickle
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# allow all urls since i am only running locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# load the saved model stuff
with open('best_model.pkl', 'rb') as f:
    best_model = cloudpickle.load(f)

with open('category_means.pkl', 'rb') as f:
    category_means = cloudpickle.load(f)

with open('most_common_values.pkl', 'rb') as f:
    most_common_values = cloudpickle.load(f)


categorical_features = ['brand', 'model', 'fuel_type', 'ext_col', 'int_col', 'engine', 'transmission']

class CarInput(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    milage: Optional[float] = None
    ext_col: Optional[str] = None
    int_col: Optional[str] = None
    accident: Optional[int] = None
    clean_title: Optional[int] = None
    hp: Optional[float] = None
    L: Optional[float] = None
    cyl_count: Optional[int] = None
    electric: Optional[int] = None
    turbo: Optional[int] = None
    trans_speed: Optional[int] = None
    manual: Optional[int] = None
    automatic: Optional[int] = None
    model_year: Optional[int] = None

# process the info, replace any missing values with the most common one.
def process_input(input_data: CarInput):
    df = {
        'brand': input_data.brand or most_common_values['brand'],
        'model': input_data.model or most_common_values['model'],
        'milage': input_data.milage or most_common_values['milage'],
        'fuel_type': most_common_values['fuel_type'],
        'engine': most_common_values['engine'],
        'transmission':most_common_values['transmission'],
        'ext_col': input_data.ext_col or most_common_values['ext_col'],
        'int_col': input_data.int_col or most_common_values['int_col'],
        'accident': input_data.accident or most_common_values['accident'],
        'clean_title': input_data.clean_title or most_common_values['clean_title'],
        'hp': input_data.hp or most_common_values['hp'],
        'L': input_data.L or most_common_values['L'],
        'cylCount': input_data.cyl_count or most_common_values['cylCount'],
        'electric': input_data.electric or most_common_values['electric'],
        'turbo': input_data.turbo or most_common_values['turbo'],
        'trans_speed': input_data.trans_speed or most_common_values['trans_speed'],
        'manual': input_data.manual or most_common_values['manual'],
        'automatic': input_data.automatic or most_common_values['automatic'],
        'model_age': 2024 - (input_data.model_year or most_common_values['model_age'])
    }
    
    # run the target encoding
    for cat in categorical_features:
        most_common = category_means[cat].get('__most_common__', 0)
        df[cat] = category_means[cat].get(df[cat], most_common)

    return pd.DataFrame([df])

# the predict route, process input data run priduction are return the prediced price.
@app.post("/predict")
async def predict_price(car_input: CarInput):
    processed_df = process_input(car_input)
    predicted_price = best_model.predict(processed_df)[0]
    return {"predicted_price": f"${predicted_price:.2f}"}


# run with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
