import os
import googleapiclient.discovery

# Fetching the data from Youtube API
def google_api(key,vidId):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = key

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="id,snippet",
        maxResults=100,
        order="relevance",
        videoId=vidId
    )
    response = request.execute()
    return response
key = "Your_API_KEY"  
vidId = "621oD2zBSbI"
response = google_api(key,vidId)
print(response)

# Creation of Data Frame
import pandas as pd 
def create_df_author_comments(response):
  authorname = []
  comments = []
  for i in range(len(response["items"])):
    authorname.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"])
    comments.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
  df_1 = pd.DataFrame(comments, index = authorname,columns=["Comments"])
  return df_1 
df = create_df_author_comments(response)
print(df)

# Cleaning Round 1
import re
def cleaning_comments(comment):
  comment = re.sub("[🤣|🤭|🤣|😁|🤭|❤️|👍|🏴|😣|😠|💪|🙏]+",'',comment)
  comment = re.sub("[0-9]+","",comment)
  comment = re.sub("[\:|\@|\)|\*|\.|\$|\!|\?|\,|\%|\"]+"," ",comment)
  return comment
df["Comments"]= df["Comments"].apply(cleaning_comments)
print(df)

# Cleaning Round 2
def cleaning_comments1(comment):
  comment = re.sub("[💁|🌾|😎|♥|🤷‍♂]+","",comment)
  comment = re.sub("[\(|\-|\”|\“|\#|\!|\/|\«|\»|\&]+","",comment)
  return comment 

df["Comments"] = df["Comments"].apply(cleaning_comments1)
print(df)

# Cleaning Round 3
def cleaning_comments3(comment):
  comment = re.sub("\n"," ",comment)
  comment = re.sub('[\'|🇵🇰|\;|\！]+','',comment)
  return comment 
df["Comments"] = df["Comments"].apply(cleaning_comments3)
print(df)

#Cleaning Round 4 
lower = lambda comment: comment.lower()
df['Comments'] = df['Comments'].apply(lower)
print(df)

# Final DataFrame after cleaning
print(df)

#Removing Empty Comments
def remove_comments(df):
  # Checks for comments which has zero length in a dataframe
  zero_length_comments = df[df["Comments"].map(len) == 0]
  # taking all the indexes of the filtered comments in a list
  zero_length_comments_index = [ind for ind in zero_length_comments.index]
  # removing those rows from dataframe whose indexes matches 
  df.drop(zero_length_comments_index, inplace = True)
  return df
df = remove_comments(df)
df

# Removing Non English Comments
from textblob.blob import TextBlob

lang_detection = lambda text: TextBlob(text).detect_language()

def remove_non_english_comments(df):
  comment = df[df["Comments"].map(lang_detection) != 'en']
  authors = [author for author in comment.index]
  df.drop(authors,inplace = True)
  return df

df = remove_non_english_comments(df)
df  

#Finding Subjectivity

def find_subjectivity_on_single_comment(text):
  return TextBlob(text).sentiment.subjectivity
   
def apply_subjectivity_on_all_comments(df):
  df['Subjectivity'] = df['Comments'].apply(find_subjectivity_on_single_comment)
  return df 

df = apply_subjectivity_on_all_comments(df)
df

#Finding Polarity

def find_polarity_of_single_comment(text):
   return  TextBlob(text).sentiment.polarity

def find_polarity_of_every_comment(df):  
  df['Polarity'] = df['Comments'].apply(find_polarity_of_single_comment)
  return df 

df = find_polarity_of_every_comment(df)
df

# Analysis Based on Polarity
analysis = lambda polarity: 'Positive' if polarity > 0 else 'Neutral' if polarity == 0 else 'Negative' 

def analysis_based_on_polarity(df):
  df['Analysis'] = df['Polarity'].apply(analysis)
  return df
  
df = analysis_based_on_polarity(df)
df

# Printing Positive Comments

def print_positive_comments():
  print('Printing positive comments:\n')
  sortedDF = df.sort_values(by=['Polarity']) 
  for i in range(0, sortedDF.shape[0] ):
    if( sortedDF['Analysis'][i] == 'Positive'):
      print(str(i+1) + '> '+ sortedDF['Comments'][i])
      print()
print_positive_comments()

# Printing Negative Comments

def print_negative_comments():
  print('Printing negative comments:\n')
  sortedDF = df.sort_values(by=['Polarity']) 
  for i in range(0, sortedDF.shape[0] ):
    if( sortedDF['Analysis'][i] == 'Negative'):
      print(str(i+1) + '> '+ sortedDF['Comments'][i])
      print()
print_negative_comments()

# Printing Neutral Comments

def print_neutral_comments():
  print('Printing neutral comments:\n')
  sortedDF = df.sort_values(by=['Polarity']) 
  for i in range(0, sortedDF.shape[0] ):
    if( sortedDF['Analysis'][i] == 'Neutral'):
      print(str(i+1) + '> '+ sortedDF['Comments'][i])
      print()
print_neutral_comments()

# Forming WordClouds

import matplotlib.pyplot as plt
from sklearn.feature_extraction import text 
from wordcloud import WordCloud

def generate_word_clouds(df):
  allWords = ' '.join([twts for twts in df['Comments']])
  wordCloud = WordCloud(stopwords = text.ENGLISH_STOP_WORDS ,width=1000, height=600, random_state=21, max_font_size=110).generate(allWords)
  plt.imshow(wordCloud, interpolation="bilinear")
  plt.axis('off')
  plt.show()

generate_word_clouds(df)
