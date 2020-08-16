from nlpcc_parser import SMUM_Data
from pandas import merge
import pandas

class Pack:
    def __init__(self):
        self.checkin = pandas.DataFrame()
        self.tweet = pandas.DataFrame()
        self.tag = pandas.DataFrame()

class DataSet:

    def __init__(self, data):
        # data should be preloaded

        self.train_male = Pack()
        self.test_male = Pack()
        self.train_female = Pack()
        self.test_female = Pack()
        self.data = data


    def build(self, p):
        profile = self.data.profile
        male = profile[profile.gender == 'm']
        female = profile[profile.gender == 'f']
        # Alloc
        train_male  =   male.iloc[:int(male.shape[0]*p),:]
        test_male   =   male.iloc[int(male.shape[0]*p):,:]
        train_female =   female.iloc[:int(female.shape[0]*p),:]
        test_female  =   female.iloc[int(female.shape[0]*p):,:]

        # Reference
        checkin     =   self.data.checkins
        tweet       =   self.data.tweets
        tag         =   self.data.tags

        # Join
        checkin_train_male   =   merge(train_male, checkin)
        checkin_test_male    =   merge(test_male, checkin)
        tweet_train_male     =   merge(train_male, tweet)
        tweet_test_male      =   merge(test_male, tweet)
        tag_train_male       =   merge(train_male, tag)
        tag_test_male        =   merge(test_male, tag)

        checkin_train_female  =   merge(train_female, checkin)
        checkin_test_female   =   merge(test_female, checkin)
        tweet_train_female    =   merge(train_female, tweet)
        tweet_test_female     =   merge(test_female, tweet)
        tag_train_female      =   merge(train_female, tag)
        tag_test_female       =   merge(test_female, tag)

        # Select
        self.train_male.checkin   =   checkin_train_male[checkin_train_male.gender == 'm']
        self.test_male.checkin    =   checkin_test_male[checkin_test_male.gender == 'm']
        self.train_male.tweet     =   tweet_train_male[tweet_train_male.gender == 'm']
        self.test_male.tweet      =   tweet_test_male[tweet_test_male.gender == 'm']
        self.train_male.tag       =   tag_train_male[tag_train_male.gender == 'm']
        self.test_male.tag        =   tag_test_male[tag_test_male.gender == 'm']

        self.train_female.checkin   =   checkin_train_female[checkin_train_female.gender == 'f']
        self.test_female.checkin    =   checkin_test_female[checkin_test_female.gender == 'f']
        self.train_female.tweet     =   tweet_train_female[tweet_train_female.gender == 'f']
        self.test_female.tweet      =   tweet_test_female[tweet_test_female.gender == 'f']
        self.train_female.tag       =   tag_train_female[tag_train_female.gender == 'f']
        self.test_female.tag        =   tag_test_female[tag_test_female.gender == 'f']
