import os
import sys
import getopt
import argparse                         
from argparse import ArgumentParser
os.system('pip install google')
                                            

def code(test_case,file_path):  
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
    
    
        


    # For cleaning the text
    from nltk.corpus import stopwords
    import regex as re
    import string



    # For building our model
    
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
            f2= open("Information.txt",'a',encoding = 'utf-8')
        try:
            f1 = open("Predictions.txt",'a',encoding = 'utf-8')
        except:
            open("Predictions.txt",'x')
            f1 = open("Predictions.txt",'a',encoding = 'utf-8')

        f2.write("\nTEST CASE-"+test_case+" "+keywords[0]+"'s sentiment classified as:\n")
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
        tweets_translated=tweets
       
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

            
        
        lst=[]
        lst_of_traits=[]
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


