# install packages
from transformers import pipeline
import pandas as pd
import os
import matplotlib.pyplot as plt

# data
file = os.path.join(os.getcwd(), "data", "fake_or_real_news.csv")
data = pd.read_csv(file, index_col=0)


# classify
classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base",
                      return_all_scores=True)


# run model on data with label of emotions with max score
max_score_df = []

# loop
for i in data[["title"]]: 
    for j in data[i]:
        emotion = classifier(j)

        for emotion_dicts in emotion:

            emotion_list = emotion_dicts #list
            
            max_score = max(emotion_list, key=lambda x:x['score'])
            max_score_df.append(max_score)


# convert max_score_df into a pandas dataframe
max_score_df = pd.DataFrame(max_score_df)

# occurrences for each emotion
plot_df =  max_score_df['label'].value_counts()
plot_df = pd.DataFrame(plot_df)
plot_df = plot_df.reset_index()


# plot
colors = {"anger": "red", "fear": "black", "joy": "yellow", "disgust": "green", 
         "neutral": "grey", "surprise": "pink", "sadness": "blue"}

# bar plot of distribution of emotions
plt.bar(plot_df["label"], plot_df["count"], color=plot_df['label'].map(colors),
        width = 0.4)
plt.xlabel("Emotion labels")
plt.ylabel("Count")
plt.title("Emotion classification for every headline in the Fake & Real news dataset", fontsize=10)
plt.suptitle("Distribution of emotions across all of the data", fontsize=14)
plt.show()
plt.savefig('out/emotion_distribution_all.png')
