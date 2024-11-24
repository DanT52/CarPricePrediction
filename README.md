# CarPricePrediction

This project was for this kaggle comptetition [Regression of Used Car Prices](https://www.kaggle.com/competitions/playground-series-s4e9/overview)

I used Random Forrest Regression, with scikit learn.

The notebook can be veiwed in the `notebook.html` file or `notebook.ipynb`, I've also exported the notebook as a python script in `carPriceRandomForest.py`

## Running the project

the python requirments are in `requirements.txt`

### Training the model

- Either run the notebook or the python script.
    - This should take 5-10 min.



### Running the interactive front end where you can input custom data

Make sure that you ahve already ran the notebook or the carPriceRandomForest.py,
as we need the pkl files they output to be in the same directory.

first go to the car-prediction-app directory

```
cd car-prediction-app
```
install requirments
```
npm install
```
run the frontend
```
npm run dev
```

### Running back end api that does the estimates
make sure that you have the required python libraries installed
```
pip install -r requirements.txt
```
run the api
```
estimateAPI.py
```