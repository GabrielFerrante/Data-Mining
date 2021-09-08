#Carregando base de textos (Twitter)

# https://www.kaggle.com/datatattle/covid-19-nlp-text-classification
!unzip tweets_corona_virus.zip


import pandas as pd
dataset = pd.read_csv('Corona_NLP_train.csv',encoding='iso8859')
dataset

dataset = dataset[(dataset.Sentiment=='Extremely Negative') | (dataset.Sentiment=='Extremely Positive')][['OriginalTweet','Sentiment']].dropna().sample(2000,random_state=42)
dataset.reset_index(inplace=True,drop=True)
dataset


#Classificador (Bidirectional GRU, Treinando a própria Embedding)

import keras
import ktrain
from ktrain import text

#Pré-processando (tente alterar os parâmetros)

(x_train, y_train), (x_test, y_test), preproc = text.texts_from_df(dataset, 
                                                                   'OriginalTweet',
                                                                   label_columns='Sentiment',
                                                                   maxlen=128, 
                                                                   max_features=1000,
                                                                   preprocess_mode='standard',
                                                                   lang=None,
                                                                   ngram_range=1,
                                                                   val_pct = 0.3,
                                                                   random_state = 42
                                                                   )

model = text.text_classifier('bigru', (x_train, y_train) , preproc=preproc)
classifier = ktrain.get_learner(model, 
                             train_data=(x_train, y_train), 
                             val_data=(x_test, y_test)
                             )

classifier.fit_onecycle(0.005,3)

classifier.validate()

#Testando para novos textos

predictor = ktrain.get_predictor(classifier.model, preproc)

predictor.get_classes()

predictor.predict('"We also have family but we can\'t stay home...Be responsible and STAY HOME')

predictor.predict_proba('"We also have family but we can\'t stay home...Be responsible and STAY HOME')