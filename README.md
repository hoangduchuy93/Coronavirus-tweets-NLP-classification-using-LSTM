![image](https://user-images.githubusercontent.com/91864024/182314544-6e928548-bf7b-459f-b413-e248edc31653.png)

# Coronavirus Tweets NLP Classification Using LSTM
## I. Outline
- Coronavirus has changed our lives and the way we live everyday. There has been a lot of loss for people and property. And we can't efford it.
- Thanks to the unprecedented strong development of vaccines, along with the awareness of vaccination and wearing masks at all times, we have partly controlled the epidemic to re-develop the economy. However, community awareness is still the top criterion to ensure safe coexistence in the context that the epidemic cannot end soon.
- This project is made using a dataset taken from twitter in 2020, during the worst covid epidemic in history. Through people's tweets, this project (for learning purposes) tries to predict their emotions to understand how they felt during that difficult time.
## II. Business Objective/ Problem
- Let's say you work in the Data Science department at Twitter. In 2020, there are many tweets every day mentioning the pandemic situation of people where they live and their feelings at that time. Your task is to build a model that predicts users' emotions through the words in their tweets.
- This project builds on that requirement
## III. Project implementation
### 1. Business Understanding
Based on the above description => identify the problem:
- Find a solution to analyze users' emotions through what they write in tweets
- Objectives/problems: build a model that can classify users' emotions through tweets. From there, understand the emotional of users, and can take actions to support them in emergancy cases.
- Applied methods:
  - Text pre-processing
  - Tokenizer, padding, LSTM model
### 2. Data Understanding/ Acquire
- The tweets have been pulled from Twitter and manual tagging has been done then.
The names and usernames have been given codes to avoid any privacy concerns. 
- You can download dataset at: https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification
- Columns:
  - UserName
  - ScreenName
  - Location
  - Tweet At
  - Original Tweet
  - Sentiment
- Dataset: there are 2 files for training and testing your model
  - Corona_NLP_train.csv: including 117184 rows
  - Corona_NLP_test.csv: including 12386 rows
 
 ![image](https://user-images.githubusercontent.com/91864024/182322557-902566ef-7d79-4ddd-8e7a-eca64a63266d.png)

### 3. Build model
#### 3.1. Understand the dataset
**- Load training and testing set**
```python
#đọc dữ liệu bằng pandas
df_train = pd.read_csv('Corona_NLP_train.csv', encoding='latin-1')
df_test = pd.read_csv('Corona_NLP_test.csv', encoding='latin-1')
```
![image](https://user-images.githubusercontent.com/91864024/182323587-d2f22a90-9819-4991-9723-dadc7e59f7bf.png)
#### 3.2. Data pre-processing
**- Select necessary features for your model**
```python
#Chọn lại data: chỉ lấy cột OriginalTweet và Sentiment
df_train = df_train[['OriginalTweet', 'Sentiment']]
df_test = df_test[['OriginalTweet', 'Sentiment']]
```
**- Check NaN, Null, duplicated for training/ seting set**
```python
#kiểm tra NaN, null, duplicate: không có
df_train.isna().sum()
df_train.isnull().sum()
df_train.duplicated().sum()
df_test.isna().sum()
df_test.isnull().sum()
df_test.duplicated().sum()
```
Comment: no NaN, Null, duplicated found
#### 3.3. Understand features
**- Check distribution in column Sentiment**
```python
plt.figure(figsize=(10, 4))
plt.pie(df_train['Sentiment'].value_counts(), labels=df_train['Sentiment'].unique(), autopct='%.1f%%',
        textprops={'color':'w'})
plt.legend(loc='upper right')
plt.axis('equal')
plt.show()
```
![image](https://user-images.githubusercontent.com/91864024/182327064-50644d07-23c3-4365-89c0-90840126e8e9.png)

Comments: Positive and Negative comments make up the majority. The remaining types have almost the same number

**- Check wordcloud in OriginalTweet each sentiment classification**
```python
from wordcloud import WordCloud
for label, cmap in zip(['Positive', 'Negative', 'Neutral', 'Extremely Positive', 'Extremely Negative'],
                       ['winter', 'autumn', 'magma', 'viridis', 'plasma']):
  text = df_train.query('Sentiment == @label')['OriginalTweet'].str.cat(sep=' ')
  plt.figure(figsize=(10, 6))
  wc = WordCloud(width=1000, height=600, background_color='#f8f8f8', colormap=cmap, max_words=30)
  wc.generate_from_text(text)
  plt.imshow(wc)
  plt.axis('off')
  plt.title(f"Words Commonly Used in ${label}$ Messages", size=20)
  plt.show()
```
![image](https://user-images.githubusercontent.com/91864024/182327914-a002f5e4-c66c-4bc5-97f4-d71c066e4af1.png)
![image](https://user-images.githubusercontent.com/91864024/182328062-32e06803-78af-4597-8494-ba98fdee6ab0.png)
![image](https://user-images.githubusercontent.com/91864024/182328091-25b40878-f287-4eef-8ad5-8c3b89869957.png)
![image](https://user-images.githubusercontent.com/91864024/182328138-b930b274-5147-445f-be7a-92deedf29ce3.png)
![image](https://user-images.githubusercontent.com/91864024/182328186-f64d30a3-84a4-49fc-9fe0-828bf3dd4d18.png)
#### 3.4. Data cleaning
**- Show some rows of tweets**
![image](https://user-images.githubusercontent.com/91864024/182328660-6e7c5d1c-cf2a-45e6-b71f-7a0a34403d7a.png)
**- Remove special characters, tokenizer, remove stopword**
```python
stopwords = nltk.corpus.stopwords.words('english')
```
**- Functions for text**
```python
# let's create some of the functions: 
punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'         # define a string of punctuation symbols

# Functions to clean tweets
def remove_links(tweet):
    """Takes a string and removes web links from it"""
    tweet = re.sub(r'http\S+', '', tweet)   # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet)  # remove bitly links
    tweet = tweet.strip('[link]')   # remove [links]
    tweet = re.sub(r'pic.twitter\S+','', tweet)
    return tweet

def remove_users(tweet):
    """Takes a string and removes retweet and @user information"""
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove re-tweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove tweeted at
    return tweet

def remove_hashtags(tweet):
    """Takes a string and removes any hash tags"""
    tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove hash tags
    return tweet

def remove_av(tweet):
    """Takes a string and removes AUDIO/VIDEO tags or labels"""
    tweet = re.sub('VIDEO:', '', tweet)  # remove 'VIDEO:' from start of tweet
    tweet = re.sub('AUDIO:', '', tweet)  # remove 'AUDIO:' from start of tweet
    return tweet

#Correct words
def split_dup_words(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"hasn't", "has not", txt)
    txt = re.sub(r"didn't", "did not", txt)
    txt = re.sub(r"wasn't", "was not", txt)
    txt = re.sub(r"weren't", "were not", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    return txt

def preprocess_tweet(tweet):
    """Main master function to clean tweets, stripping noisy characters, and tokenizing use lemmatization"""
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = remove_hashtags(tweet)
    tweet = remove_av(tweet)
    tweet = tweet.lower()  # lower case
    tweet = split_dup_words(tweet) #sửa đổi các từ viết tắt
    tweet = re.sub('[' + punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    tweet = re.split('\W+', tweet)
    tweet = [word for word in tweet if word not in stopwords]

    return tweet
 ```
 **- Map our target with labels**
 ```python
 target_mapping={'Extremely Negative':0, 'Negative':0, 'Neutral':1,
                'Positive':2, 'Extremely Positive':2}

df_train['SentimentMapped']=df_train['Sentiment'].map(lambda x:target_mapping[x])
df_test['SentimentMapped']=df_test['Sentiment'].map(lambda x:target_mapping[x])
```
![image](https://user-images.githubusercontent.com/91864024/182329464-0b08d563-09ff-4712-a0cd-40ae40b2e004.png)
**- Apply functions**
```python
df_train['OriginalTweetClean'] = df_train['OriginalTweet'].apply(lambda x: preprocess_tweet(x))
df_test['OriginalTweetClean'] = df_test['OriginalTweet'].apply(lambda x: preprocess_tweet(x))
```
![image](https://user-images.githubusercontent.com/91864024/182329761-bce90077-1e6a-4023-b3f1-e1922209d795.png)
#### 3.5. Split training/ testing set
```python
X_train = df_train['OriginalTweetClean']
X_test = df_test['OriginalTweetClean']

y_train = df_train['SentimentMapped']
y_test = df_test['SentimentMapped']
```

























