from flask import Flask,jsonify,request
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

psm = PorterStemmer()
app = Flask(__name__)


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    arr = []
    for i in text:
        if i.isalnum():
            arr.append(i)

    text = arr[:]
    arr.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            arr.append(i)

    text = arr[:]
    arr.clear()

    for i in text:
        arr.append(psm.stem(i))

    return " ".join(arr)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    input_msg = request.form.get('input_msg')
    transformed_sms = transform_text(input_msg)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]


    if result == 1:
        return jsonify({'SMS_msg': 'SPAM'})
    else :
        return jsonify({'SMS msg':'NOT SPAM'})

if __name__ == '__main__':
    app.run(debug=True)
