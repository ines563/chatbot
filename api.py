import nltk #"importation de package de langage naturel"
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() #"definir un variable pour mettre a chaque fois une mot similaire"
import pickle #"implémente des protocoles binaires pour sérialiser et désérialiser une structure d'objet " 
import numpy as np
from keras.models import load_model
model = load_model('chatbot_model.h5') #"charge
import json #" pour la stockage de données"
import random 
intents = json.loads(open('intents.json',encoding='utf-8').read()) #"lire leLe fichier d intents qui contient toutes les données que nous allons utiliser pour former le modèle"
words = pickle.load(open('words.pkl','rb')) #" charger et ouvrir le fichier qui contient tous les mots uniques qui constituent le vocabulaire de notre modèle"
classes = pickle.load(open('classes.pkl','rb'))
#Python libraries that we need to import for our bot
import random
from flask_cors import CORS #"partage de ressources cross-origin (CORS)"
from flask import Flask, request
app = Flask(__name__)
CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

#"divisié la phrase et cherche la similaire de mot "
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words] #"racine de chaque mot "
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
#retour du tableau de mots: 0 ou 1 pour chaque mot du sac qui existe dans la phrase
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence) #" donner la phrase pour le tokenisation
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

    
def get_response(msg):
    from keras.models import load_model
    model = load_model('chatbot_model.h5')
    import json
    import random
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl','rb'))
    classes = pickle.load(open('classes.pkl','rb'))
    p = bow(msg, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    print(return_list)
    ints = return_list
    tag = ints[0]['intent']
    list_of_intents = intents['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result



"""

from flask_cors import cross_origin

#We will receive messages that api sends our bot at this endpoint
 
@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def receive_message():
    if request.method == 'GET':
        return "welcome api chatbot"
    #if the request was not get, it must be POST and we can just proceed with sending a message back to user
    else:
    
       output = request.get_json()
       print(output)
       print(output["word"])
       reponse = get_response(output["word"])

    return reponse
"""



from flask import Flask, request, jsonify, make_response




@app.route("/", methods=["POST", "OPTIONS"])
def api_create_order():
    if request.method == "OPTIONS": # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST": # The actual request following the preflight
        output = request.get_json()
        reponse = get_response(output["word"])
        dicts = {}
        dicts.update({"res" : reponse} )
        return _corsify_actual_response(jsonify(dicts))
    else :
        raise RuntimeError("Weird - don't know how to handle method {}".format(request.method))

def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    app.run()
