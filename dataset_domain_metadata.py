from trainer import Trainer
import constants

import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
from pynif import NIFCollection
import re
from top2vec import Top2Vec
import dataset_utils as utils

#institute_uri = "https://orcid.org/0000-0002-1825-0097"
institute_uri = "https://anonymiz.ed/"

#dir_data = "./data/"
dir_data = "/mnt/webscistorage/wf7467/linker_topic_bias/linker_topic_analysis/data/"
dir_data_grouped = dir_data + "grouped_data/"
path_out_context_topics = "/mnt/webscistorage/wf7467/linker_topic_bias/linker_topic_analysis/data/context_topics_domain_metadata.ttl"

random.seed(42)
np.random.seed(42)
#folder_nif_datasets = "/mnt/webscistorage/wf7467/agnos/data/Generated_Datasets/license_ok"
COMPUTE_OR_LOAD = False



#print(dir(Top2Vec))
topic_model = Top2Vec.load(constants.model_path)
path_topic_names = constants.path_topic_names
dict_topic_names = utils.get_topic_names(path_topic_names=None)#path_topic_names)

new_collection = NIFCollection(uri=institute_uri)
topic_lines = []

for tpl_doc in utils.get_all_text_documents():
    input_document = tpl_doc[0]
    dataset = tpl_doc[1]
    doc_key = tpl_doc[2]
    topics_words, word_scores, topic_scores, topic_nums = topic_model.query_topics(query=input_document, num_topics=1, reduced=False, tokenizer=utils.tok)
    topic_num = topic_nums[0]

    # Get existing one or instantiate a new one

    # Add context to new collection
    context_uri = f"{institute_uri}{dataset}/{doc_key}"
    new_context = new_collection.add_context(uri=context_uri, mention=input_document)

    topic_name = dict_topic_names.get(topic_num, "None")#
    sanitized_topic_name = utils.sanitize_filename(topic_name)

    # Adding lines for nif:topic afterwards
    topic_lines.append((context_uri, sanitized_topic_name, topic_name, topic_num, dataset))


generated_nif = new_collection.dumps(format='turtle')
nif_dataset_output_path = path_out_context_topics#f"{dir_data_grouped}domain_metadata.nif"
with open(nif_dataset_output_path, "w", encoding='utf-8') as dataset_file:
    dataset_file.write(generated_nif)
print("Saved successfully to", nif_dataset_output_path)



with open(dir_data+"nif_topics.tmp.nif", "w", encoding='utf-8') as dataset_file:
    dataset_file.write(utils.generate_nif_topics_str(topic_lines))
    #for topic_line in topic_lines:
    #    dataset_file.write(f'<{topic_line[0]}> nif:topic "{topic_line[1]}" .\n\n')

print("Saved domain-grouped collection successfully.")