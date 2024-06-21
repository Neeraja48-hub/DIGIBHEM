import nltk
import json

nltk.download('punkt')

def load_movie_lines(file_name):
    lines = open(file_name, encoding='utf-8', errors='ignore').read().split('\n')
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    return id2line

def load_conversations(file_name, id2line):
    conv_lines = open(file_name, encoding='utf-8', errors='ignore').read().split('\n')
    conversations = []
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')
        conv = _line[-1][1:-1].replace("'", "").replace(" ", "")
        conv = conv.split(',')
        for i in range(len(conv) - 1):
            conversations.append((id2line[conv[i]], id2line[conv[i+1]]))
    return conversations

def preprocess_sentence(sentence):
    return ' '.join(nltk.word_tokenize(sentence.lower()))

def prepare_data(conversations):
    input_texts = []
    target_texts = []
    for conv in conversations:
        input_texts.append(preprocess_sentence(conv[0]))
        target_texts.append(preprocess_sentence(conv[1]))
    return input_texts, target_texts

# Load and preprocess data
id2line = load_movie_lines('cornell_movie_dialogs_corpus/movie_lines.txt')
conversations = load_conversations('cornell_movie_dialogs_corpus/movie_conversations.txt', id2line)
input_texts, target_texts = prepare_data(conversations)

# Save the processed data
data = {'input_texts': input_texts, 'target_texts': target_texts}
with open('processed_data.json', 'w') as f:
    json.dump(data, f)

print("Data preprocessing completed.")
