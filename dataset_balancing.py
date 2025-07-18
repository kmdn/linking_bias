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
from pynif import NIFCollection, NIFContext, NIFPhrase
import nltk
from top2vec import Top2Vec
import glob
import dataset_utils as utils


#institute_uri = "https://orcid.org/0000-0002-1825-0097"
institute_uri = "https://anonymiz.ed/"

#dir_data = "./data/"
dir_data = "/mnt/webscistorage/wf7467/linker_topic_bias/linker_topic_analysis/data/"
dir_data_grouped = dir_data + "grouped_data/"

random.seed(42)
np.random.seed(42)
#folder_nif_datasets = "/mnt/webscistorage/wf7467/agnos/data/Generated_Datasets/license_ok"
COMPUTE_OR_LOAD = False

nltk.download('stopwords')

#print(dir(Top2Vec))
topic_model = Top2Vec.load(constants.model_path)
#path_topic_names = "/mnt/webscistorage/wf7467/linker_topic_bias/linker_topic_analysis/topic_names_distiluse-base-multilingual-cased_36_topics.txt"
dict_topic_names = utils.get_topic_names(path_topic_names=constants.path_topic_names)
#dict_topic_names = utils.get_topic_names(path_topic_names=None)


nif_files = glob.glob(dir_data + "/*.nif")
print(f"NIF files found using glob: {nif_files}")

# Keep all collections in a dictionary, where the key is the topic number
dict_collections = {}

for file in nif_files:
    print(f"Processing file: {file}")
    collection = utils.load_nif_dataset(file)
    for context in collection.contexts:
        input_document = context.mention
        #print(f"Processing context: {context.uri} with mention: {input_document}")
        #topic_num = -1#get topic number for the document
        topics_words, word_scores, topic_scores, topic_nums = topic_model.query_topics(query=input_document, num_topics=1, reduced=False, tokenizer=utils.tok)
        topic_num = topic_nums[0]
        #print(f"--> Topic[{topic_num}]")
        #print(f"Topic[{topic_num}] for input document '{input_document}': {topics_words[0]} with score {topic_scores[0]}")
        #print(f"Topic number for document '{input_document}': {topic_num[0][0]} with score {topic_num[1][0]}")
        # Get existing one or instantiate a new one
        new_collection = dict_collections.get(topic_num, 
                                              NIFCollection(uri=institute_uri+"domain"+str(topic_num)+"/"))
        dict_collections[topic_num] = new_collection
        # Add context to new collection
        new_context = new_collection.add_context(uri=context.uri, mention=context.mention)

        # Copy all mentions linked to this context...
        for e in context.phrases:
            # asc_order_mentions_updated: (mention, beginIndex, new_mention, original_entity, new_entity)
            # Add an entry
            phrase = new_context.add_phrase(
            
            beginIndex=e.beginIndex,
            endIndex=e.endIndex,
            annotator=e.annotator,
            score=e.score,
            taIdentRef=e.taIdentRef,
            taIdentRefLabel=e.taIdentRefLabel,
            taClassRef= e.taClassRef,
            taMsClassRef= e.taMsClassRef,
            uri = e.uri,
            #is_hash_based_uri= e.is_hash_based_uri,
            )


for topic_num, collection in dict_collections.items():
    topic_name = dict_topic_names.get(topic_num, "None")#
    sanitized_topic_name = utils.sanitize_filename(topic_name)
    # Save each collection to a separate file
    #print(f"Saving collection for topic {topic_num} with {len(collection.contexts)} contexts.")
    generated_nif = collection.dumps(format='turtle')
    nif_dataset_output_path = f"{dir_data_grouped}domain_{topic_num}_{sanitized_topic_name}.nif"

    print(f"Domain[{topic_num}]: {len(collection.contexts)} contexts, saving to {nif_dataset_output_path}")

    with open(nif_dataset_output_path, "w", encoding='utf-8') as dataset_file:
        dataset_file.write(generated_nif)
    print("Saved successfully to", nif_dataset_output_path)

print("All collections saved successfully.")


#for topic_num, collection in dict_collections.items():
#    print(f"Domain[{topic_num}]: {len(collection.contexts)}")

len_sum = 0
min_contexts_len = len(dict_collections[0].contexts) if dict_collections else 0
min_contexts_idx = 0

# Go through all contexts to see the amount of contexts per domain
for topic_num in range(len(dict_collections)):
    print(f"Domain[{topic_num}]: {len(dict_collections[topic_num].contexts)}")
    len_contexts = len(dict_collections[topic_num].contexts)
    len_sum += len_contexts
    if len_contexts < min_contexts_len:
        min_contexts = len_contexts
        min_contexts_idx = topic_num


print(f"Minimum contexts found in domain {min_contexts_idx} with {min_contexts_len} contexts.")
# Choose which contexts aka. input sentences for each domain we want to use
chosen_contexts = {}
chosen_contexts_uri = {}
for topic_num in range(len(dict_collections)):
    print(f"Domain[{topic_num}]: {len(dict_collections[topic_num].contexts)}")
    for i in range(min_contexts_len):
        # Take this many context samples from each domain
        cont = True
        choosing_counter = 0
        while cont:
            choosing_counter += 1
            if choosing_counter > 1000:
                raise RuntimeError(f"Warning: Too many attempts to choose a context for topic {topic_num}.")

            # Randomly choose a context from the current domain
            # If the context has already been chosen, choose another one
            # If not, add it to the chosen contexts
            chosen_context: NIFContext = random.choice(dict_collections[topic_num].contexts)
            # Check that the current one has not been chosen yet
            already_chosen_uris = chosen_contexts_uri.get(topic_num, [])
            # initialise with empty list if not yet set and add this context and context URI
            if len(already_chosen_uris) == 0:
                # Add the URI
                chosen_contexts_uri[topic_num] = [chosen_context.uri]
                # Add the context
                chosen_contexts[    topic_num] = [chosen_context    ]
                cont = False
                continue

            # Check if the chosen context URI is already in the list of chosen URIs for this topic
            if chosen_context.uri not in already_chosen_uris:
                chosen_contexts_uri[topic_num].append(chosen_context.uri)
                chosen_contexts[    topic_num].append(chosen_context    )
                cont = False
            else:
                # This context has already been chosen, so we skip it and...
                # Retry the randomness...
                pass
        

balanced_collection = NIFCollection(uri=f"{institute_uri}/domain/balanced/")
# Keep track of topics and add them separately to the collection file because pynif does not support nif:topic
topic_lines = []
for topic_num, contexts in chosen_contexts.items():
    print(f"Chosen contexts for Domain[{topic_num}]: {len(contexts)}")
    # Print the URIs of the chosen contexts
    for context in contexts:
        print(f"  - {context.uri}")

        new_context = balanced_collection.add_context(uri=context.uri, mention=context.mention)
        # Copy all mentions linked to this context...
        for e in context.phrases:
            # asc_order_mentions_updated: (mention, beginIndex, new_mention, original_entity, new_entity)
            # Add an entry
            phrase = new_context.add_phrase(
                beginIndex=e.beginIndex,
                endIndex=e.endIndex,
                annotator=e.annotator,
                score=e.score,
                taIdentRef=e.taIdentRef,
                taIdentRefLabel=e.taIdentRefLabel,
                taClassRef= e.taClassRef,
                taMsClassRef= e.taMsClassRef,
                uri = e.uri,
            )
        
        # Keeping track of lines for nif:topic afterwards
        topic_name = dict_topic_names.get(topic_num, "None")
        sanitized_topic_name = utils.sanitize_filename(topic_name)
        topic_lines.append((new_context.uri, sanitized_topic_name, topic_name, topic_num, "balanced_collection"))

nif_topics = utils.generate_nif_topics_str(topic_lines)
generated_nif = balanced_collection.dumps(format='turtle')
nif_dataset_output_path = f"{dir_data}DOMiNO.nif"
with open(nif_dataset_output_path, "w", encoding='utf-8') as dataset_file:
    dataset_file.write(generated_nif)
print("Saved balanced collection successfully to", nif_dataset_output_path)


# Now append the nif:topic lines to the end of the file
with open(nif_dataset_output_path, "a", encoding='utf-8') as dataset_file:
    # Add some newline for better readability
    dataset_file.write("\n\n\n")
    dataset_file.write(nif_topics)
print(f"NIF topics appended successfully to {nif_dataset_output_path}.")

#dict_topic_names

print("Total number of contexts across all domains:", len_sum)