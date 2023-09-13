import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch
from keras_tuner import Objective
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer


train = pd.read_csv("/Users/ttonny0326/GitHub_Project/LSTM_wikipedia/train.csv", encoding='ISO-8859-1', low_memory=False)
test_labels = pd.read_csv("/Users/ttonny0326/GitHub_Project/LSTM_wikipedia/test_labels.csv", encoding='ISO-8859-1', low_memory=False)
test = pd.read_csv("/Users/ttonny0326/GitHub_Project/LSTM_wikipedia/train.csv", encoding='ISO-8859-1', low_memory=False)

stopwords = set(stopwords.words('english'))
nltk.download('stopwords')


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    #re.sub()依據pattern及repl對string進行處理，結果回傳處理過的新字串，將前面 "" 中的文字轉換成後面 "" 中的文字 

    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text) 
    #移除非文字字元，並將其轉為空格 

    text = re.sub('[^A-Za-z\' ]+', '',text) 
    #除了單引號的非字母、數字字元，其餘皆刪除並且保留單引號，刪掉其他符號，例如@#$%^&*@&#^@&

    text = re.sub('\s+', ' ', text)
    #將任何非空白字元皆轉為空格，換行符號轉換為空格，例如\n 轉為空格

    text = text.strip(' ')
    #strip() 方法用於移除字串頭尾指定的字符（默認為空格或換行符）或字符序列，將首尾空格刪掉，例如123 abc改為123abc

    text = ' '.join([word for word in text.split() if word not in (stopwords)]) 
    #將評論中的定冠詞去除，例如the 刪掉

    return text
    #統整國外相關處理方式


train["comment_text"] = train["comment_text"].fillna('').apply(clean_text)
test["comment_text"] = test["comment_text"].fillna('').apply(clean_text)

#使用上方自訂函數導入評論進行處理
#使用fillna('')將缺失值替換成空格''


train_data = train["comment_text"]
test_data = test["comment_text"]
#將處理好的評論給定名稱

train_label=train[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']]
test_label=test_labels[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']]
#將六個labels裁好給定另一個名稱

tokenizer = Tokenizer(num_words = 40000) #40000 words are used here
tokenizer.fit_on_texts(train_data)

#建立一個40000的token字典

train_final = tokenizer.texts_to_sequences(train_data) #train_final儲存每個評論的word轉為的數字list
test_final = tokenizer.texts_to_sequences(test_data)


totalNumWords = [len(one_comment) for one_comment in train_final ] #totalNumWords儲存每個評論的字個數
plt.hist(totalNumWords,bins = np.arange(0,410,10))
plt.title("Distribution of Word Nos.")
plt.ylabel('No. of comments', fontsize=12)
plt.xlabel('No. of words in each comments', fontsize=12)
plt.show
#橫軸為:每個評論的word數
#縱軸為:評論個數

train_padded =pad_sequences(train_final, maxlen=150)
test_padded =pad_sequences(test_final, maxlen=150)
print("Shape of training data",train_padded.shape)
print("Shape of testing data",test_padded.shape)
#進行截長補短，讓所有評論的數字 list 長度都改為150


def build_model(hp):
    model = keras.Sequential()
    # Embedding layer
    model.add(layers.Embedding(
        input_dim=40000,
        output_dim=hp.Int('embedding_output_dim', min_value=32, max_value=128, step=16),
        input_length=150
    ))
    # First LSTM layer
    model.add(layers.LSTM(
        units=hp.Int('lstm_units_1', min_value=32, max_value=128, step=16),
        dropout=hp.Float('lstm_dropout_1', min_value=0.0, max_value=0.5, step=0.1),
        return_sequences=True
    ))
    # Second LSTM layer
    model.add(layers.LSTM(
        units=hp.Int('lstm_units_2', min_value=32, max_value=128, step=16),
        dropout=hp.Float('lstm_dropout_2', min_value=0.0, max_value=0.5, step=0.1)
    ))
    # Dense layer
    model.add(layers.Dense(
        units=6,
        activation='sigmoid'
    ))
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(
                  hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy',
                  metrics=['AUC'])
    
    return model

x_train, x_val, y_train, y_val = train_test_split(train_padded, train_label, shuffle = True, random_state = 123)

# tuner = RandomSearch(
#     build_model,
#     objective=Objective("val_auc", direction="max"),
#     max_trials=5,  # Number of model configurations to try
#     executions_per_trial=1,  # Number of times to fit each model (can be used to average results if desired)
#     directory='keras_tuner_dir',
#     project_name='lstm_tuning'
# )

# tuner.search_space_summary()

# tuner.search(x_train, y_train,
#             epochs=3,
#             validation_data=(x_val, y_val),
#             batch_size = 70
# )

# Build the model with the best hyperparameters
model = Sequential()
model.add(Embedding(input_dim=40000, output_dim=96, input_length=150))  
model.add(LSTM(units=96, dropout=0.1, return_sequences=True))
model.add(LSTM(units=48, dropout=0.2))
model.add(Dense(units=6, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate = 0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])

# Check the model architecture
model.summary()

# # find out the best value of batch size and epoch
# for batch_size in [32,50,64,70,128]:
#     for epochs in [1,2,3]:  
#         model = Sequential()
#         model.add(Embedding(input_dim=40000,output_dim=64, input_length=150))  
#         model.add(LSTM(units = 64, dropout = 0.2,return_sequences=True))
#         model.add(LSTM(units = 64, dropout = 0.2))
#         model.add(Dense(units = 6, activation = 'sigmoid'))
#         model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["AUC"])
#         x_train, x_val, y_train, y_val = train_test_split(train_padded, train_label, shuffle = True, random_state = 123)
#         #對參數每種組合，訓練一個SVC
#         history=model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_val, y_val))
#         #用驗證資料集(valid set)評估SVC

# Train the model after finding all the best hyperparameter 
x_train, x_val, y_train, y_val = train_test_split(train_padded, train_label, shuffle=True, random_state=123)
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3, batch_size=70)


plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.history['auc'], label='Training AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('Training and Validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.show()


# Save the tokenizer
# In this case, saving the tokenizer is essential due to the reason that this project was using the comment from the Internet
# Therefore, if any user tend to use this model, they must use the same data set or data pool from the train data
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


model.save('lstm_model.h5')


### Data preprocessing / train-test split / the function of model-building for Random Search(the value of the model itself)
### Model compile / Model fit / Result visualization / Model saving