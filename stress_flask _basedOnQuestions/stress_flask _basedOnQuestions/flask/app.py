from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
from spacy.lang.en import English
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

punctuations = string.punctuation
stopwords = set(STOP_WORDS)
parser = English()
nlp = spacy.load("en_core_web_sm")


app = Flask(__name__)
app.secret_key = "secret key"

model = load_model(filepath="model/ABC_model.h5")

with open(file="model/tokens.pkl", mode="rb") as file:
    tok = pickle.load(file=file)

class_names = {0: 'Stress', 1: 'Non-Stress'}

not_stop = ["aren't", "are", "couldn't", "could", "didn't", "did", "doesn't", "does", "don't", "do", "hadn't", "had",
            "hasn't", "has", "haven't", "have", "isn't", "is", "mightn't", "might", "mustn't", "must", "needn't",
            "need", "no", "nor", "not", "shan't", "shouldn't", "should", "wasn't", "was", "weren't", "were", "wouldn't",
            "would"]

for i in not_stop:
    if i in stopwords:
        stopwords.remove(i)
    else:
        continue

cList = pickle.load(open('model/cword_dict.pkl', 'rb'))
c_re = re.compile('(%s)' % '|'.join(cList.keys()))


def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group()]
    return c_re.sub(replace, text)


def preprocess_text(docx):
    sentence = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in docx]
    sentence = [word for word in sentence if word not in stopwords and word not in punctuations]
    sentence = [word for word in sentence if len(word) > 1 and word.isalpha()]
    sentence = list(set(sentence))
    return sentence


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        q1 = request.form['q1']
        q2 = request.form['q2']
        q3 = request.form['q3']
        q4 = request.form['q4']
        q5 = request.form['q5']

        df = pd.DataFrame(data=[str(q1), str(q2), str(q3), str(q4), str(q5)])

        df.columns = [["text"]]

        sentences = []
        for sentes in df["text"].values:
            sentes = expandContractions(str(sentes))
            sp_sentes = nlp(text=sentes)
            cleaned = preprocess_text(sp_sentes)
            sentences.append(cleaned)
        num_data = tok.texts_to_sequences(sentences)
        pad_text = pad_sequences(sequences=num_data, maxlen=300, padding="post")
        model_prediction = model.predict(pad_text)

        model_pred = []
        for i in range(len(model_prediction)):
            model_pred.append(np.argmax(model_prediction[i]))
        print(len(model_pred))
        print(model_pred)
        stress_score = model_pred.count(1) * 20
        print(stress_score)
        prediction = ''
        if stress_score < 50:
            prediction = "stress"
            print("predicted as stress")
        elif 60 <= stress_score < 80:
            prediction = "neutral"
            print("predicted as neutral")
        elif stress_score >= 80:
            prediction = "non-stress"
            print("predicted as non-stress")
        else:
            print("something went wrong")

        return render_template("index1.html", prediction=prediction, acc=stress_score)



@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        graph_name = request.form['text']
        graph = ''
        name = ''
        if graph_name == "classification1":
            name = "Adaboost Classification Report"
            graph = "static/graphs/adaboost_classificationreport.PNG"
        elif graph_name == 'classification2':
            name = "RandomForest Classification Report"
            graph = "static/graphs/randomforest_classificationreport.PNG"
        elif graph_name == 'confusion1':
            name = "Adaboost Confusionmatrix"
            graph = "static/graphs/adaboost_confusionmatrix.png"
        elif graph_name == 'confusion2':
            name = "RandomForest Confusionmatrix"
            graph = "static/graphs/randomforest_confusionmatrix.png"
        elif graph_name == 'accuracy':
            name = "Accuracy Comparison Graph"
            graph = "static/graphs/accuracy_comparision.png"
        elif graph_name == 'pieChart':
            name = "Visualization by PieChart"
            graph = "static/graphs/pie.png"
        elif graph_name == 'barChart':
            name = "Visualization by BarChart"
            graph = "static/graphs/hbar.png"


        return render_template('graphs.html', name=name, graph=graph)


@app.route('/back', methods=['POST', 'GET'])
def back():
    return render_template('index1.html')


@app.route("/")
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form["email"]
        pwd = request.form["password"]
        r1 = pd.read_excel('user.xlsx')
        for index, row in r1.iterrows():
            if row["email"] == str(email) and row["password"] == str(pwd):
                return render_template("index1.html")
                # return redirect(url_for('home'))
        else:
            msg = 'Invalid Login Try Again'
            return render_template('login.html', msg=msg)
    return render_template('login.html')


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['Email']
        password = request.form['Password']
        col_list = ["name", "email", "password"]
        r1 = pd.read_excel('user.xlsx', usecols=col_list)
        new_row = {'name': name, 'email': email, 'password': password}
        r1 = r1.append(new_row, ignore_index=True)
        r1.to_excel('user.xlsx', index=False)
        print("Records created successfully")
        # msg = 'Entered Mail ID Already Existed'
        msg = 'Registration Successful !! U Can login Here !!!'
        return render_template('login.html', msg=msg)
    return render_template('login.html')


@app.route('/password', methods=['POST', 'GET'])
def password():
    if request.method == 'POST':
        current_pass = request.form['current']
        new_pass = request.form['new']
        verify_pass = request.form['verify']
        r1 = pd.read_excel('user.xlsx')
        for index, row in r1.iterrows():
            if row["password"] == str(current_pass):
                if new_pass == verify_pass:
                    r1.replace(to_replace=current_pass, value=verify_pass, inplace=True)
                    r1.to_excel("user.xlsx", index=False)
                    msg1 = 'Password changed successfully'
                    return render_template('password_change.html', msg1=msg1)
                else:
                    msg2 = 'Re-entered password is not matched'
                    return render_template('password_change.html', msg2=msg2)
        else:
            msg3 = 'Incorrect password'
            return render_template('password_change.html', msg3=msg3)
    return render_template('password_change.html')


@app.route('/graphs', methods=['POST', 'GET'])
def graphs():
    return render_template('graphs.html')


# @app.route('/confusion', methods=['POST', 'GET'])
# def confusion():
#     return render_template('graphs.html')
#
#
# @app.route('/classy', methods=['POST', 'GET'])
# def classy():
#     return render_template('graphs.html')


@app.route('/logout', methods=['POST', 'GET'])
def logout():
    return render_template('login.html')


if __name__ == '__main__':
    app.run(port=5002, debug=True, threaded=True)
