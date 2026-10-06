from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI()

# Load models
with open("delay_model.pkl", "rb") as f:
    delay_model = pickle.load(f)

with open("crowd_model.pkl", "rb") as f:
    crowd_model = pickle.load(f)

class PredictionRequest(BaseModel):
    time_of_day: str
    day_of_week: str
    transit_line: str
    weather: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    return Path("index.html").read_text(encoding="utf-8")

@app.post("/api/predict")
def predict(request: PredictionRequest):
    # Convert request to DataFrame
    df = pd.DataFrame([{
        "time_of_day": request.time_of_day,
        "day_of_week": request.day_of_week,
        "transit_line": request.transit_line,
        "weather": request.weather
    }])
    
    # Predict
    delay_pred = delay_model.predict(df)[0]
    crowd_label = crowd_model.predict(df)[0]
    
    # Process delay
    delay_minutes = int(round(max(0, min(delay_pred, 60))))
    delay_bar_pct = int(round((delay_minutes / 60) * 100))
    
    if delay_minutes <= 2:
        delay_verdict = 'On time — no disruption expected.'
    elif delay_minutes <= 7:
        delay_verdict = 'Minor delay — trains running slightly behind schedule.'
    elif delay_minutes <= 15:
        delay_verdict = 'Moderate delay — plan extra buffer time.'
    else:
        delay_verdict = 'Significant delay — consider alternate routes or timing.'
        
    # Process crowding
    crowd_pct_map = {
        'Minimal': 10,
        'Low': 25,
        'Moderate': 50,
        'High': 75,
        'Extreme': 95
    }
    crowd_pct = crowd_pct_map.get(crowd_label, 50)
    
    if crowd_pct < 30:
        crowd_verdict = 'Plenty of space — comfortable journey likely.'
    elif crowd_pct < 60:
        crowd_verdict = 'Moderate occupancy — seating available.'
    elif crowd_pct < 85:
        crowd_verdict = 'Crowded — standing room, some discomfort.'
    else:
        crowd_verdict = 'Extremely crowded — expect difficulty boarding.'
        
    return {
        "delay_minutes": delay_minutes,
        "delay_bar_pct": delay_bar_pct,
        "delay_verdict": delay_verdict,
        "crowding_label": crowd_label,
        "crowding_pct": crowd_pct,
        "crowd_verdict": crowd_verdict
    }

# Ensure the server can be run with uvicorn
# Command: uvicorn api:app --reload
