import re
import nltk
import pickle
from nltk.corpus import stopwords
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# Assuming the Tokenizer was fit on the entire dataset before saving the model
# tokenizer = Tokenizer(num_words=40000)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocess the text (using the previously defined clean_text function)
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

    text = ' '.join([word for word in text.split() if word not in (stop_words)]) 
    #將評論中的定冠詞去除，例如the 刪掉

    return text

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'] # You can adjust this threshold as needed

def predict_topic(model, content):
    cleaned_content = clean_text(content)
    sequence = tokenizer.texts_to_sequences([cleaned_content])
    padded_sequence = pad_sequences(sequence, maxlen=150)
    
    prediction = model.predict(padded_sequence)[0]  # Extracting the prediction list from its wrapping list
    threshold = 0.5
    for label, prob in zip(LABELS, prediction):
        if prob >= threshold:
            print(f"{label}: {prob:.2f} -> This is classified as {label} comment")
        else:
            print(f"{label}: {prob:.2f} -> Not {label} enough")
    print("\n")

if __name__ == "__main__":
    model_path = "lstm_model.h5" 
    model = load_model(model_path)
    while True:
        try:
            text = input("Enter text (or type 'exit' to quit): ")
            if text.lower() == 'exit':
                break
            predict_topic(model, text)
        except Exception as e:
            print(f"Error: {e}")







