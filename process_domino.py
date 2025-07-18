from pynif import NIFCollection
from pynif import NIFContext, NIFPhrase
from top2vec import Top2Vec
import gensim
import nltk

stopwords = nltk.corpus.stopwords.words('english')

def load_nif_dataset(nif_data_path):
    print(f"Loading NIF dataset... [{nif_data_path}]")
    # Load the dataset
    nif_data = ""
    parsed_collection = None
    with open(nif_data_path, 'r', encoding="utf-8") as f:
        nif_data = f.read()
        parsed_collection = NIFCollection.loads(nif_data, format='turtle')
    print("Finished reading NIF data from file(%s)." % nif_data_path)
    print(parsed_collection)
    #parsed_collection = load_nif_dataset(nif_data_path = "C:\\Users\\wf7467\\Desktop\\Evaluation Datasets\\Datasets\\entity_linking\\conll_aida-yago2-dataset\\AIDA-YAGO2-dataset.tsv_nif")
    return parsed_collection

def remove_stopwords(text):
    text = [word for word in text if word not in stopwords]
    return text

def tok(text):
    text = gensim.utils.simple_preprocess(text)
    text = remove_stopwords(text)
    return text



domino_path = "/mnt/webscistorage/wf7467/linker_topic_bias/linker_topic_analysis/data/DOMiNO_no_topics.nif"
model_path = '/mnt/webscistorage/wf7467/linker_topic_bias/unjde/analysis/model/topic2vec'
collection = load_nif_dataset(nif_data_path=domino_path)
topic_model = Top2Vec.load(model_path)


for context in collection.contexts:
    input_document = context.mention
    #print(f"Processing context: {context.uri} with mention: {input_document}")
    #topic_num = -1#get topic number for the document
    topics_words, word_scores, topic_scores, topic_nums = topic_model.query_topics(query=input_document, num_topics=1, reduced=False, tokenizer=tok)
    topic_num = topic_nums[0]

    # Get the F1 score for the topic and generate F1 score table
    print("Input document:", input_document)
    print("Topic number:", topic_num)
