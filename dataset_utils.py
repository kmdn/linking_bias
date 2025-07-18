import gensim
from trainer import Trainer
import constants
import glob
import json
import re
from pynif import NIFCollection
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random


def get_topic_names(path_topic_names=None, total_topic_count=37):
    if path_topic_names is None:
        raise ValueError("Path to topic names file must be provided.")
    dict_topic_names = {}
    with open(path_topic_names, "r") as f:
        lines_topic_names = f.readlines()
        for line in lines_topic_names: 
            for i in range(total_topic_count):
                prefix_topic_line = f"Topic {i}:"
                if prefix_topic_line in line:
                    #print(f"Topic {i}: {line}")
                    topic_name = line.split(":")[1].replace("*", "").replace('"', "").strip()
                    print(f"Topic {i}: {topic_name}")
                    dict_topic_names[i] = topic_name
                    break
                else:
                    if f"Topic {i}:**" in line:
                        print(f"Could not find topic name for topic {i} in line: {line.strip()}")
                    #print(f"Skipping line: {line.strip()}")
                    pass

#     if path_topic_names is None:
#         topics_names = \
# """Medical Research Methods
# Conflict & Political Violence
# Governance & Administration
# Financial Market Insights
# Corporate Announcements
# Sports Coaching & Management
# Football League Analysis
# Cricket Performance Metrics
# Chinese Sociopolitical Landscape
# Athletic Achievements & Championship
# German Language Constructs
# Commodity Trading Dynamics
# Financial Performance Metrics
# News Outlets & Reporting
# Notable Athletes & Celebrities
# Stock Market Insights
# Corporate Collaborations
# Political Campaigns & Elections
# Sports League Standings
# International Soccer Competitions
# Sports Highlights and Matches
# Baseball and Player Profiles
# Kurdish Political Landscape
# Soccer Leagues and Matches
# Tennis Tournaments and Champions
# Research and Reports
# Baseball Inning Details
# Football and Player Profiles
# Miscellaneous German Phrases
# Tennis Tournament Highlights
# MLB Teams and Matchups
# Israeli-Palestinian Relations
# Soccer Leagues and Competitions
# Soccer and Player Profiles
# MLB Team Rivalries"""
#         topics_names = topics_names.split("\n")
#         dict_topic_names = {i: topic_name for i, topic_name in enumerate(topics_names) if topic_name.strip()}


    return dict_topic_names



def sanitize_filename(filename):
    # Replace spaces with underscores
    filename = re.sub(r'[ ]', '_', filename)
    # Remove all non-alphanumeric characters except underscores, hyphens, and dots
    filename = re.sub(r'[^a-zA-Z0-9_\-.]', '', filename)
    return filename

def remove_stopwords(text):
    text = [word for word in text if word not in constants.stopwords]
    return text
def tok(text):
    text = gensim.utils.simple_preprocess(text)
    text = remove_stopwords(text)
    return text


# Load NIF datasets
# for each document, check the topic, group it by topic and output into a separate file
def load_nif_dataset(nif_data_path = "./data/ACE2004N.nif"):
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

def get_embeddings_and_labels():
    embeddings = {}
    text_labels = []
    raw_text_labels_files = []

    for file_path in constants.file_paths:
        with open(file_path + "_embeddings_.json", "r") as f:
            embeddings_objects = json.load(f)
            for emb_obj in embeddings_objects:
                embeddings[emb_obj["hash"]] = np.array(emb_obj["embeddings"])

        with open(file_path + "_labeled_fewer_classes.json", "r") as f:
            text_labels_file = json.load(f)
            raw_text_labels_files.append({'ds': file_path, 'raw': text_labels_file})
            for text_hash, doc_info in text_labels_file.items():
                labels_list = [label["system"] for label in doc_info["label"]]
                text_labels.append((text_hash, labels_list))
    return embeddings, text_labels, raw_text_labels_files


def get_all_text_documents():
    embeddings, text_labels, raw_text_labels_files = get_embeddings_and_labels()
    #text_labels[0]
    # one entry per dataset for raw_text_labels_files
    dataset_idx = 0
    for dataset_idx in range(len(raw_text_labels_files)):
        dataset = raw_text_labels_files[dataset_idx]['ds']
        dataset_name = dataset.split("/")[-1].split(".")[0]
        #KORE_50_DBpedia.ttl

        print("Dataset: ", dataset)
        # one entry per document for raw_text_labels_files[0]['raw']
        # raw_text_labels_files[0]['raw']
        doc_keys = list(raw_text_labels_files[dataset_idx]['raw'].items())
        print(type(doc_keys))

        for document_idx in range(len(doc_keys)):
            doc_key = doc_keys[document_idx][0]
            print("Document key: ",doc_key)
            input_document = raw_text_labels_files[dataset_idx]['raw'][doc_key]['doc']
            yield (input_document, dataset_name, doc_key)


def generate_nif_topics(topic_lines):
    nif_topics = []
    for topic_line in topic_lines:
        context_uri, sanitized_topic_name, topic_name, topic_num, dataset = topic_line
        nif_topics.append(f'<{context_uri}> nif:topic "{sanitized_topic_name}" .\n')
    return nif_topics

def generate_nif_topics_str(topic_lines):
    nif_topics = generate_nif_topics(topic_lines)
    nif_topics_str = ""
    for line in nif_topics:
        nif_topics_str += line.strip()+"\n"
    return nif_topics_str
