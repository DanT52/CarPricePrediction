from fastapi.testclient import TestClient
from estimateAPI import app

client = TestClient(app)

def test_predict_price():
    sample_input = {
        "brand": "Toyota",
        "model": "Camry",
        "milage": 100000,
        "ext_col": "White",
        "int_col": "Black",
        "accident": 0,
        "clean_title": 1,
        "hp": 200,
        "L": 2.5,
        "cyl_count": None,
        "electric": 0,
        "turbo": 0,
        "trans_speed": 6,
        "manual": 0,
        "automatic": 1,
        "model_year": 2018
    }

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    response_json = response.json()
    assert "predicted_price" in response_json
    print(f"Predicted price: {response_json['predicted_price']}")

if __name__ == "__main__":
    test_predict_price()
    print("Test passed!")