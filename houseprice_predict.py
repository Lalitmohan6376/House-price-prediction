from flask import Flask,render_template,url_for,request,jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))

@app.route("/")
def home():
    return render_template("front_page.html")

@app.route("/predict", methods=["POST"])
def predict():

    area_type = request.form['area_type']
    availability = request.form['availability']
    bhk = int(request.form['size'])
    location = request.form['location']
    total_sqft = float(request.form['total_sqft'])
    bath = float(request.form['bath'])
    balcony = float(request.form['balcony'])

    # Step 1: create dataframe 
    input_df = pd.DataFrame([{
        'area_type': area_type,
        'availability': availability,
        'BHK': bhk,
        'location': location,
        'total_sqft': total_sqft,
        'bath': bath,
        'balcony': balcony
    }])

    # Step 2: get_dummies
    input_df = pd.get_dummies(
        input_df,
        columns=['area_type','availability','location']
    )

    # Step 3: columns align karo
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Step 4: scaling
    num_cols = ['total_sqft', 'bath', 'balcony']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Step 5: prediction
    prediction = model.predict(input_df)[0]

    return render_template(
        "front_page.html",
        prediction_text=f"Estimated House Price: ₹ {round(prediction,2)} Lakhs"
    )


if __name__ == '__main__':
    app.run(debug=True)