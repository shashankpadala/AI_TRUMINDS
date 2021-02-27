import os
import sys
import getopt
import argparse                         
from argparse import ArgumentParser
os.system('pip install sentencepiece')
os.system('pip install google')
import sentencepiece
os.system('pip install wget')
import wget


filename = wget.download('https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py')                                                       

def code(test_case,file_path):
    try:
        import tensorflow
    except:
        os.system('pip install tensorflow')    
    try:
        import warnings
        warnings.filterwarnings("ignore")
    except:
        os.system('pip install warn')
        warnings.filterwarnings("ignore")
    try:
        from googlesearch import search
    except:
        os.system('pip install google')
    try:
        from googletrans import Translator
    except:
        os.system('pip install googletrans==3.1.0a0')
    try:
        from gensim.models import Word2Vec,KeyedVectors
    except:
        os.system('pip install gensim')
    try:
        import nltk
    except:
        os.system('pip install nltk')
        import nltk
    try:
        import tweepy
    except:
        os.system('pip install tweepy')
    try:
        from sklearn.model_selection import train_test_split, GridSearchCV
    except:
        os.system('pip install scikit-learn')
    try:
        import pandas
    except:
        os.system('pip install pandas')
    try:
        import numpy
    except:
        os.system('pip install numpy')
    try:
        import pickle
    except:
        os.system('pip install pickle-mixin')
    try:
        import spacy
    except:
        os.system('pip install spacy')
    try:
        import regex
    except:
        os.system('pip install regex')
    try:
        import string
    except:
        os.system('pip install strings')           
    try:
        import regex
    except:
        os.system('pip install regex')
    try:
        import keras
    except:
        os.system('pip install keras')
    try:
        import tokenization
    except:
        os.system('pip install tokenization')
    try:
        import tensorflow_hub as hub
    except:
        os.system('pip install tensorflow-hub')
    
        


    # For cleaning the text
    import spacy
    import tensorflow_hub as hub
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import regex as re
    import string



    # For building our model
    import tensorflow as tf
    import tensorflow.keras
    import sklearn
    from tensorflow.keras import layers, Model
    from tensorflow.keras.optimizers import SGD, Adam
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D
    import tokenization
    import nltk
    from googlesearch import search
    import tweepy
    import nltk
    from googletrans import Translator
    import numpy as np 
    import pandas as pd 
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.ensemble import BaggingClassifier
    import pickle
    from gensim.models import Word2Vec,KeyedVectors
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC



    def authe():
        consumer_key='XTHhMymaYDmr7sQgH1AwFYLmz'
        consumer_secret='HDExd4a6jEYf8Q7LkS7W322YsdOf6leYj9XUfcs96ljTjMszcO'
        access_token='1023046632034643969-VDQoi159Bktz5s2OVIa36x4CRAIJyu'
        access_secret='hhaHqxrCIdTblV9VVMv38tXRXp0ywOCpgggBrBZcTmDGB'
        auth=tweepy.OAuthHandler(consumer_key=consumer_key,consumer_secret=consumer_secret)
        auth.set_access_token(access_token,access_secret)
        return auth
    def get_twitter_client():
        auth=authe()
        client=tweepy.API(auth,wait_on_rate_limit=True)
        return client

    def vector_for_tweet(tweets):
        tweets_vec=[nltk.word_tokenize(i) for i in tweets]
        model=Word2Vec(tweets_vec,min_count=1,size=300)
        final_data=[]
        for i in range(len(tweets)):
            lst=[]
            words = nltk.word_tokenize(tweets[i])
            max_arr = max_fun(words, model)
            min_arr = min_fun(words, model)
            avg_arr = avg_fun(words, model)
            lst.extend(max_arr)
            lst.extend(min_arr)
            lst.extend(avg_arr)
            final_data.append(lst)
        return final_data


    def max_fun(words, model):
        max_arr=[]
        for i in range(300):
            max = model[words[0]][i]
            for j in range(len(words)):
                if(model[words[j]][i]>max):
                    max = model[words[j]][i]
            max_arr.append(max)
        return max_arr

    def min_fun(words, model):
        min_arr=[]
        for i in range(300):
            min = model[words[0]][i]
            for j in range(len(words)):
                if(model[words[j]][i]<min):
                    min = model[words[j]][i]
            min_arr.append(min)
        return min_arr

    def avg_fun(words, model):
        avg_arr=[]
        for i in range(300):
            count=0
            for j in range(len(words)):
                count+=model[words[j]][i]
            avg_arr.append(count/len(words))
        return avg_arr

    def count_ones_and_zeros_percent(arr):
        z=0
        o=0
        for i in arr:
            if i==0:
                z+=1
            else:
                o+=1
        ones_per = o/len(arr)
        return ones_per*100


    class TweetClassifier:
    
        def __init__(self, tokenizer, bert_layer, max_len, lr = 0.0001,
                     epochs = 15, batch_size = 32,
                     activation = 'sigmoid', optimizer = 'SGD',
                     beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                     metrics = 'accuracy', loss = 'binary_crossentropy'):
            
            self.lr = lr
            self.epochs = epochs
            self.max_len = max_len
            self.batch_size = batch_size
            self.tokenizer = tokenizer
            self.bert_layer = bert_layer
            

            self.activation = activation
            self.optimizer = optimizer
            
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.epsilon =epsilon
            
            self.metrics = metrics
            self.loss = loss

            
        def encode(self, texts):
            
            all_tokens = []
            masks = []
            segments = []
            
            for text in texts:
                
                tokenized = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]'])
                
                len_zeros = self.max_len - len(tokenized)
                
                
                padded = tokenized + [0] * len_zeros
                mask = [1] * len(tokenized) + [0] * len_zeros
                segment = [0] * self.max_len
                
                all_tokens.append(padded)
                masks.append(mask)
                segments.append(segment)
            
            return np.array(all_tokens), np.array(masks), np.array(segments)


        def make_model(self):
            
            # Shaping the inputs to our model
            
            input_ids = Input(shape = (self.max_len, ), dtype = tf.int32, name = 'input_ids')
            
            input_mask = Input(shape = (self.max_len, ), dtype = tf.int32, name = 'input_mask')
            
            segment_ids = Input(shape = (self.max_len, ), dtype = tf.int32,  name = 'segment_ids')

            
            pooled_output, sequence_output = bert_layer([input_ids, input_mask, segment_ids] )



            clf_output = sequence_output[:, 0, :]
            
            out = tf.keras.layers.Dense(1, activation = self.activation)(clf_output)
            
            
            model = Model(inputs = [input_ids, input_mask, segment_ids], outputs = out)
            
            # define the optimizer

            if self.optimizer is 'SGD':
                optimizer = SGD(learning_rate = self.lr)

            elif self.optimizer is 'Adam': 
                optimizer = Adam(learning_rate = self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)

            model.compile(loss = self.loss, optimizer = self.optimizer, metrics = [self.metrics])
            
            print('Model is compiled with {} optimizer'.format(self.optimizer))
            
            return model
        
        
        
        
        def train(self, x):    
            
            checkpoint = ModelCheckpoint('model.h5', monitor='val_loss',
                                         save_best_only=True)
                
            
            model = self.make_model()
            
            X = self.encode(x['cleaned_text'])
            Y = x['cNEU']
            
            model.fit(X, Y, shuffle = True, validation_split = 0.2, 
                      batch_size=self.batch_size, epochs = self.epochs,
                      callbacks=[checkpoint])
                    
            print('Model is fit!')
            
                
        def predicto(self, x):
            
            X_test_encoded = self.encode(x['STATUS'])
            best_model = tf.keras.models.load_model('pickle_models/model_o.h5',custom_objects={'KerasLayer':hub.KerasLayer})
            y_pred = best_model.predict(X_test_encoded) 
            return y_pred
        def predictc(self, x):
            
            X_test_encoded = self.encode(x['STATUS'])
            best_model = tf.keras.models.load_model('pickle_models/model_c.h5',custom_objects={'KerasLayer':hub.KerasLayer})
            y_pred = best_model.predict(X_test_encoded) 
            return y_pred
        def predicte(self, x):
            
            X_test_encoded = self.encode(x['STATUS'])
            best_model = tf.keras.models.load_model('pickle_models/model_e.h5',custom_objects={'KerasLayer':hub.KerasLayer})
            y_pred = best_model.predict(X_test_encoded) 
            return y_pred
        def predicta(self, x):
            
            X_test_encoded = self.encode(x['STATUS'])
            best_model = tf.keras.models.load_model('pickle_models/model_a.h5',custom_objects={'KerasLayer':hub.KerasLayer})
            y_pred = best_model.predict(X_test_encoded) 
            return y_pred
        def predictn(self, x):
            
            X_test_encoded = self.encode(x['STATUS'])
            best_model = tf.keras.models.load_model('pickle_models/model_n.h5',custom_objects={'KerasLayer':hub.KerasLayer})
            y_pred = best_model.predict(X_test_encoded) 
            return y_pred


    def tokenize_tweets(text_):
        return tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(text_) + ['[SEP]'])


    


    #taking file name as input
    f_name = file_path
    o = open(f_name,'r')
    info = o.read()
    o.close()
    users_list=info.split("\n")
    for user in users_list:
        keywords=user.split(",")
        query=""
        for key in keywords:
            query=query+key
        query=query+'  twitter'

        try:
            f2 = open("Information.txt",'a',encoding = 'utf-8')
        except:
            open("Information.txt",'x')
            f2 = open("Information.txt",'a',encoding = 'utf-8')
        try:
            f1 = open("Predictions.txt",'a',encoding = 'utf-8')
        except:
            open("Predictions.txt",'x')
            f1 = open("Predictions.txt",'a',encoding = 'utf-8')

        f2.write("\nTEST CASE-"+test_case+"  "+keywords[0]+"'s sentiment classified as:\n")
        f1.write("\nTEST CASE-"+test_case+" Personality of "+keywords[0]+" predicted is \n")

        
        for j in search(query,tld='co.in', num=1, stop=1, pause=2):         
            twitter_source=j
        temp1=twitter_source.split('/')
        user_id=temp1[3].split('?')[0]
        client=get_twitter_client()
        tweets=[]
        try:
            for status in tweepy.Cursor(client.user_timeline, id=user_id).items(500):
                tweets.append(status.text)
        except:
            for i in tweepy.Cursor(client.search_users,q=user_id,count=1).items(1):
                user_id=i.screen_name
            for status in tweepy.Cursor(client.user_timeline, id=user_id).items(500):
                tweets.append(status.text)
        tweets_translated=[]

        for i in tweets:
            translator = Translator()
            tweets_translated.append(translator.translate(i,dest='en').text)
        
        f2.write("URL from which subjects data extracted is: https://www.twiiter.com/"+user_id+"\n")
        
        

        with open('pickle_models/hate_forest_final.pickle', 'rb') as f:
            hate_model = pickle.load(f)
        with open('pickle_models/hate_vector_final.pickle', 'rb') as f:
            hate_vectorizer = pickle.load(f)
        tweets_hate=hate_vectorizer.transform(tweets_translated)
        hate_predicted=hate_model.predict(tweets_hate)
        hate_tweets=0
        for i in hate_predicted:
            if (i==1):
                hate_tweets=hate_tweets+1
        hate_tweets_percent=(hate_tweets/len(hate_predicted))*100        #hate percent

        if(hate_tweets_percent > 5):
            f2.write("Individual hates others mostly and the percentage of hating tweets is "+str(hate_tweets_percent)+".\n")
        else:
            f2.write("Individual doesn't hates others mostly and the percentage of hating tweets is "+str(hate_tweets_percent)+".\n")


        with open('pickle_models/clf_sex.pickle', 'rb') as f:
            sexist_model = pickle.load(f)
        sexism_predicted=sexist_model.predict(tweets_translated)
        sexism_tweets=0
        for i in sexism_predicted:
            if (i==1):
                sexism_tweets=sexism_tweets+1
        sexism_tweets_percent=(sexism_tweets/len(sexism_predicted))*100        #sexism percent

        if(sexism_tweets_percent > 5):
            f2.write("Individual promotes sexism and has a percentage of tweets related to it is "+str(sexism_tweets_percent)+".\n")
        else:
            f2.write("Individual doesn't promotes sexism and has a percentage of tweets related to it is "+str(sexism_tweets_percent)+".\n")


        

        
        with open('pickle_models/bullying_model_file', 'rb') as f:
            bullying_model = pickle.load(f)
        with open('pickle_models/countvector_bullying_model_file', 'rb') as f:
            bullying_countvector = pickle.load(f)
        tweets_cvb=bullying_countvector.transform(tweets_translated)
        bullying_predicted=bullying_model.predict(tweets_cvb)
        bullying_tweets=0
        non_bullyin_tweets=0
        for i in bullying_predicted:
            if (i==1):
                bullying_tweets=bullying_tweets+1
        bullying_tweets_percent=(bullying_tweets/len(bullying_predicted))*100                #bullying_percent

        if(bullying_tweets_percent > 1):
            f2.write("The individual has more bullying personality against others and the percentage of bullying tweets is "+str(bullying_tweets_percent)+".\n")
        else:
            f2.write("The individual has less bullying personality against others and the percentage of bullying tweets is "+str(bullying_tweets_percent)+".\n")

            
        
            #USING BERT
        lst=[]
        lst_of_traits=[]
        try:
            text =tweets_translated
            bert=pd.DataFrame(text,columns=['STATUS'])

            BERT_MODEL_HUB = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
            bert_layer = hub.KerasLayer(BERT_MODEL_HUB, trainable=True)
            to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
            vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
            FullTokenizer = tokenization.FullTokenizer
            tokenizer = FullTokenizer(vocabulary_file, to_lower_case)
            bert['tokenized_tweets'] = bert.STATUS.apply(lambda x: tokenize_tweets(x))


            max_len = len(max(bert.tokenized_tweets, key = len))
            bert['padded_tweets'] = bert.tokenized_tweets.apply(lambda x: x + [0] * (194-len(x)))   
            final_data=bert.drop(['tokenized_tweets','padded_tweets'],axis=1)
            classifier = TweetClassifier(tokenizer = tokenizer, bert_layer = bert_layer,
                                          max_len = 194, lr = 0.0001,
                                          epochs = 5,  activation = 'sigmoid',
                                          batch_size = 32,optimizer = 'SGD',
                                          beta_1=0.9, beta_2=0.999, epsilon=1e-07)
            
            o=count_ones_and_zeros_percent(np.round(classifier.predicto(final_data)))
            count=o
            if count>50:
              lst_of_traits.append("Openness trait is high")
            else:
              lst_of_traits.append("Openness trait is less")
            c=count_ones_and_zeros_percent(np.round(classifier.predictc(final_data)))
            count=c
            if count>50:
              lst_of_traits.append("Conscientiousness trait is high")
            else:
              lst_of_traits.append("Conscientiousness trait is Less")
            e=count_ones_and_zeros_percent(np.round(classifier.predicte(final_data)))
            count=e
            if count>50:
              lst_of_traits.append("Extraversion trait is high")
            else:
              lst_of_traits.append("Extraversion trait is Less")
            a=count_ones_and_zeros_percent(np.round(classifier.predicta(final_data)))
            count=a
            if count>50:
              lst_of_traits.append("Agreeableness trait is high")
            else:
              lst_of_traits.append("Agreeableness trait is Less")
            n=count_ones_and_zeros_percent(np.round(classifier.predictn(final_data)))
            count=n
            if count>50:
              lst_of_traits.append("Neuroticism trait is high")
            else:
              lst_of_traits.append("Neuroticism trait is Less")
            ######
        except Exception as e:
            print(e)
            text =tweets_translated 
            New_vectors=vector_for_tweet(text)
            infile = open('pickle_models/O_model.sav','rb')
            o = pickle.load(infile)
            infile = open('pickle_models/C_model.sav','rb')
            c = pickle.load(infile)
            infile = open('pickle_models/E_model.sav','rb')
            e = pickle.load(infile)
            infile = open('pickle_models/A_model.sav','rb')
            a = pickle.load(infile)
            infile = open('pickle_models/N_model.sav','rb')
            n = pickle.load(infile)

            y_pred=o.predict(New_vectors)
            count=count_ones_and_zeros_percent(y_pred)
            if count>50:
                lst_of_traits.append("Openness trait is high")
            else:
                lst_of_traits.append("Openness trait is less")
            y_pred=c.predict(New_vectors)
            count=count_ones_and_zeros_percent(y_pred)
            if count>50:
                lst_of_traits.append("Conscientiousness trait is high")
            else:
                lst_of_traits.append("Conscientiousness trait is Less")
            y_pred=e.predict(New_vectors)
            count=count_ones_and_zeros_percent(y_pred)
            if count>50:
                lst_of_traits.append("Extraversion trait is high")
            else:
                lst_of_traits.append("Extraversion trait is Less")
            y_pred=a.predict(New_vectors)
            count=count_ones_and_zeros_percent(y_pred)
            if count>50:
                lst_of_traits.append("Agreeableness trait is high")
            else:
                lst_of_traits.append("Agreeableness trait is Less")
            y_pred=n.predict(New_vectors)
            count=count_ones_and_zeros_percent(y_pred)
            if count>50:
                lst_of_traits.append("Neuroticism trait is high")
            else:
                lst_of_traits.append("Neuroticism trait is Less")

            
        with open('pickle_models/personel_attacks_model_file', 'rb') as f:
            attack_model = pickle.load(f)
        attack_predicted=attack_model.predict(tweets_translated)
        attacks=0
        for i in attack_predicted:
            if i:
                attacks=attacks+1
        attack_percent=(attacks/len(attack_predicted))*100         #attack_percent

        if(attack_percent >= 0.1):
            f2.write("Individual seems to attack others mostly and the percentage of attacking tweets is "+str(attack_percent)+".\n")
        else:
            f2.write("Individual doesn't seems to attack others mostly and the percentage of attacking tweets is "+str(attack_percent)+".\n")
        
            
        


        for i in range(len(lst_of_traits)):
            f1.write(lst_of_traits[i]+"\n")
        f2.close()
        f1.close()
        print("The information and prediction files are updated...!!!!")
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", dest="filename", required=True,
                    metavar="FILE")
    parser.add_argument("--testcase", dest="testcase", required=True,
                    )
    args = parser.parse_args()
    code(args.testcase,args.filename)


