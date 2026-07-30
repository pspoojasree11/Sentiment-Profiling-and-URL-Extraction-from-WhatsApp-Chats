!pip install emoji urlextract
# nltk pandas numpy matplotlib seaborn wordcloud
#!pip install emoji
import regex
import re
import pandas as pd
import numpy as np
import emoji
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from urlextract import URLExtract
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from urllib.parse import urlparse
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


# All Functions being used in the notebook
# Check if it is a message
def no_content(line):
    if ("<Media omitted>" in line or "This message was deleted" in line):
      return True

# Extract the Date time
def date_time(s):
    pattern = '^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)[ ]?(AM|PM|am|pm)? -'
    result = regex.match(pattern, s)
    return result

# Check existence of author
def find_author(s):
    s = s.split(":")
    if len(s) == 2:
        return True
    return False

# Extract Message
def getMessage(line):
    splitline = line.split(' - ')
    datetime = splitline[0]
    date, time = datetime.split(', ')
    message = " ".join(splitline[1:])

    if find_author(message):
        splitmessage = message.split(": ")
        author = splitmessage[0]
        message = splitmessage[1]
    else:
        author = None
    return date, time, author, message

def split_count(text):
    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(emoji.is_emoji(char) for char in word):
            emoji_list.append(word)

    return emoji_list

# Overall Score of a dataframe
def score(a,b,c):
    if (a>b) and (a>c):
        return("Positive ")
    if (b>a) and (b>c):
        return("Negative")
    if (c>a) and (c>b):
        return("Neutral")

def overallSentiment(data):
    x = sum(data["Positive"])
    y = sum(data["Negative"])
    z = sum(data["Neutral"])

    return score(x,y,z)

def emojisInChat(data):
    total_emojis_list = list([a for b in data.Emoji for a in b])
    emoji_dict = dict(Counter(total_emojis_list))
    emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
    return [i for i in emoji_dict[:10]]

data = []
conversation = 'chate.txt'
with open(conversation, encoding="utf-8") as fp:
    fp.readline()
    messageBuffer = []
    date, time, author = None, None, None
    while True:
        line = fp.readline()
        if not line:
            break
        line = line.strip()
        if no_content(line):
          continue
        if date_time(line):
            if len(messageBuffer) > 0:
                if (author is not None):
                  data.append([date, time, author, ''.join(messageBuffer)])
            messageBuffer.clear()
            date, time, author, message = getMessage(line)
            messageBuffer.append(message)
        else:
            messageBuffer.append(line)


df = pd.DataFrame(data, columns=["Date", "Time", "Author", "Message"])
df['Date'] = pd.to_datetime(df['Date'],format='%d/%m/%Y')


df["Emoji"] = df["Message"].apply(split_count)
emojis = sum(df['Emoji'].str.len())
df.head(10)

data = df.dropna()

sentiments = SentimentIntensityAnalyzer()

data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Message"]]
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Message"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Message"]]

data.head(10)

weeks = {
0 : 'Monday',
1 : 'Tuesday',
2 : 'Wednesday',
3 : 'Thrusday',
4 : 'Friday',
5 : 'Saturday',
6 : 'Sunday'
}
data['Day'] = data['Date'].dt.weekday.map(weeks)

data = data[['Date','Day','Time','Author','Message', 'Emoji', 'Positive', 'Negative', 'Neutral']]

data['Day'] = data['Day'].astype('category')

data['Letter'] = data['Message'].apply(lambda s : len(s))

data['Word'] = data['Message'].apply(lambda s : len(s.split(' ')))

URLPATTERN = r'(https?://\S+)'
data['urlcount'] = data.Message.apply(lambda x: regex.findall(URLPATTERN, x)).str.len()
links = np.sum(data.urlcount)

data.head(10)

# Create sender counts as a series
sender_count_series = df["Author"].value_counts()
sender_count_series



# Select the top 15 senders
top_senders = sender_count_series.head(5)
sns.barplot(x=top_senders.index, y=top_senders.values)
plt.xticks(rotation=90)  # Rotate x-axis labels by 90 degrees
plt.title('Top most Active members',fontdict={'fontsize': 20,'fontweight': 8})
plt.show(5)

sender_count_series = df["Author"].value_counts()
sender_count_series
top_senders = sender_count_series.head(5)
plt.pie(top_senders.values, labels= top_senders.index, autopct = "%0.2f")
plt.title('Top most active members',fontdict={'fontsize': 20,'fontweight': 8})
plt.show()

emojis = []
users = sender_count_series.index

# Changed 'message' to 'Message' to match the column name in the DataFrame
for message in df['Message']:
    emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
a = pd.DataFrame(Counter(emojis).most_common(50))
a

'''
text = " ".join(review for review in data.Message)
print(f"There are {len(text)} words in all the messages.")
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
'''
text = " ".join(review for review in data.Message)
print(f"There are {len(text)} words in all the messages.")
stopwords = set(STOPWORDS)

text = re.sub(r'[^\w\s]', '', text).lower()

if len(text.split()) > 0:
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
else:
    print("No words left to generate wordcloud after removing stopwords and punctuation.")

# Most Active Author in the chat
plt.figure(figsize=(9,6))
author_value_counts = data['Author'].value_counts()
most_active = author_value_counts.iloc[0:20]
most_active.plot.barh()

plt.xlabel('No. of messages',fontdict={'fontsize': 14,'fontweight': 10})
plt.ylabel('Authors',fontdict={'fontsize': 14,'fontweight': 10})
plt.title('Mostly active member of chat',fontdict={'fontsize': 20,'fontweight': 8})
plt.show()

plt.figure(figsize=(8,5))
t = data['Time'].value_counts().head(20)
tx = t.plot.bar()

plt.xlabel('Time',fontdict={'fontsize': 12,'fontweight': 10})
plt.ylabel('No. of messages',fontdict={'fontsize': 12,'fontweight': 10})
plt.title('Analysis of time when chat was highly active.',fontdict={'fontsize': 18,'fontweight': 8})
plt.show()

plt.figure(figsize=(8,5))
data['Date'].value_counts().head(15).plot.bar()
plt.xlabel('Date',fontdict={'fontsize': 14,'fontweight': 10})
plt.ylabel('No. of messages',fontdict={'fontsize': 14,'fontweight': 10})
plt.title('Analysis of Date on which chat was highly active',fontdict={'fontsize': 18,'fontweight': 8})
plt.show()

plt.figure(figsize=(8,5))
data['Day'].value_counts().plot.bar()
plt.xlabel('Day',fontdict={'fontsize': 14,'fontweight': 10})
plt.ylabel('No. of messages',fontdict={'fontsize': 14,'fontweight': 10})
plt.title('Analysis of Day of the Week on which chat was highly active',fontdict={'fontsize': 18,'fontweight': 8})
plt.show()

data['Day'].value_counts()

import calendar

df['month'] = df['Date'].dt.month
monthData = df['month'].value_counts()

# Get month names and sort by month number
month_names = [calendar.month_abbr[i] for i in range(1, 13)]  # Get abbreviated month names
sorted_month_data = monthData.sort_index()  # Sort by month number (index)

# Print month data in order
for month_num, count in sorted_month_data.items():
    month_name = month_names[month_num - 1]  # Get month name using month number
    print(f"{month_name}: {count}")

import calendar

timeline = df.groupby([df['Date'].dt.year, df['Date'].dt.month]).count()['Message']
print(timeline)

# Create 'year' and 'month' columns
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month_name()  # Use dt.month_name() for month names

# Create a countplot with month names as hue
sns.countplot(x='year', hue='month', data=df,
              hue_order=calendar.month_name[1:])  # Order hue by month names

authorData = pd.DataFrame(columns=['Author', 'Total Messages', 'Sentiment', 'Average Words', 'Total Links'])

l = data.Author.unique()
for i in range(len(l)):
    req_df = data[data["Author"] == l[i]]

    words_per_message = (np.sum(req_df['Word']))/req_df.shape[0]

    w_p_m = ("%.3f" % round(words_per_message, 2))

    links = sum(req_df["urlcount"])

    authorData.loc[i] = [l[i], req_df.shape[0], overallSentiment(req_df), w_p_m, links]

# Dataframe stores stats of each author
authorData.head(10)

total_messages = data.shape[0]
links = np.sum(data.urlcount)

print('Group Chat Stats : ')
print(f'Total Number of Messages : {total_messages}')
print(f'Total Number of Links : {links}')
print(f'Overall Sentiment : {overallSentiment(data)}')
print(f'Favourite Emojis : {emojisInChat(data)}')

# To display the stats
for i in range(len(authorData[:10])):
    print(f"Stats for {authorData.iloc[i].Author}:")
    print(f"Messages sent: {authorData.iloc[i]['Total Messages']}")
    print(f"Average Words per Message: {authorData.iloc[i]['Average Words']}")
    print(f"Links sent: {authorData.iloc[i]['Total Links']}")
    print(f"Overall Sentiment: {authorData.iloc[i]['Sentiment']}\n")

# Initialize URL extractor and sentiment analyzer
extractor = URLExtract()
sentiments = SentimentIntensityAnalyzer()

extractor = URLExtract()

links = []
# Change 'message' to 'Message' to match the column name in the DataFrame
for message in df['Message']:
    links.extend(extractor.find_urls(message))

plt.figure(figsize=(12, 16))

# Get the domain counts and order them
# Assuming 'links' contains URLs extracted from messages
df['Domain'] = df['Message'].str.extract(r'https?://(?:www\.)?([^/]+)')  # Extract domain
# ... (rest of your plotting code)
domain_counts = df['Domain'].value_counts()
top_domains = domain_counts.index

# Create a color palette with a gradient
num_colors = len(top_domains)
palette = sns.color_palette("viridis", n_colors=num_colors)  # You can choose other palettes too

# Create the countplot with custom colors
ax = sns.countplot(y='Domain', data=df, order=top_domains, palette=palette)

plt.title('Count of Each Site URL Shared')
plt.xlabel('Count')
plt.ylabel('Site URL')

max_count = df['Domain'].value_counts().max()
# Set a desired upper limit for the x-axis
desired_upper_limit = 50  # Replace with your desired value

plt.xlim(0, desired_upper_limit)

plt.show()
