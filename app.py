import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # dump(request.form.values())
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]

    data={"Medications":request.form['Medications'],"Previous Diagnoses":request.form['PreviousDiagnoses'],"Anxiety Diagnosis":request.form['AnxietyDiagnosis'],"Obsession Type":request.form['ObsessionType'],"Education Level":request.form['EducLevel']}
    df = pd.DataFrame(data,index=[0])
    prediction = model.predict(df)


    output = round(prediction[0], 2)
    outputSentence="une depression" if output==1 else "aucune depression"
    return render_template('index.html', prediction_text=f"Le patient a   {outputSentence}")



if __name__ == "__main__":
    app.run(debug=True)