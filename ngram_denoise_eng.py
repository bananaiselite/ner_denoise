from utils import is_english, is_number, get_ngram_dict, char_compare, replace_multiple_spaces
import pandas as pd
from collections import defaultdict

def process_eng_ner(ner, char_freq_dict, min_window, max_window, threshold=1.2, stopwords=["the", "of", "an"]):
    
    if not ner or not isinstance(ner, list):
        return ner
    
    ner_lst = [n.strip() for n in ner]
    new_ner_lst = []

    for volcab in ner_lst:
        
        if is_english(volcab):
            
            words = volcab.split(" ")
            candidate_chars = []
            max_len = min(max_window, len(words))
                
            for win in range(min_window, max_len+1):
                for i in range(len(words) - win + 1):
                    char = " ".join(words[i:i+win]).lower()

                    if is_number(char) or len(char.split(" ")) < min_window or char in stopwords:
                        continue

                    candidate_chars.append(char)

            final_char, max_score = char_compare(candidate_chars, char_freq_dict)
             

            if max_score >= threshold:
                new_ner_lst.append(final_char)
        else:
            new_ner_lst.append(volcab)

    new_ner_lst = list(set(new_ner_lst))
    
    return ",".join(new_ner_lst)

def get_english_ngrams(ner_list, min_window=1, max_window=15):

    eng_prd_list = [ner.strip().lower() for ner in ner_list if is_english(ner)]
    
    char_freq_dict = get_ngram_dict(eng_prd_list, min_window=min_window, max_window=max_window, split_char=" ")
    
    filtered_dict = {k: v for k, v in char_freq_dict.items() if not (is_number(k) or len(k) == 0)}
    
    filtered_dict = dict(sorted(filtered_dict.items(), key=lambda x: x[1], reverse=True))
    
    return filtered_dict

def ner_list_sum(ner_list):
    
    ner_list = ','.join(ner_list)
    ner_list = ner_list.split(',')
    
    return ner_list

def clean(ner_list, threshold=1.2, min_window=1, max_window=10, stopwords=["the", "of", "an"]):
    ner_list = list(map(replace_multiple_spaces, ner_list))
    char_freq_dict = get_english_ngrams(ner_list = ner_list_sum(ner_list),
                                        min_window=min_window,
                                        max_window=max_window)
    result = []
    
    for ner in ner_list:
        ner = ner.split(',')
        result.append(process_eng_ner(ner=ner, 
                                     min_window=min_window,
                                     max_window=max_window, 
                                     char_freq_dict=char_freq_dict, 
                                     threshold=threshold, 
                                     stopwords=stopwords))
        
    return result

# df = pd.read_csv("NER_cls.csv")
# df = df[['content','topic_1','topic_2','NER_ORG', 'NER_FAC', 'NER_WORK_OF_ART','NER_PRODUCT']]

# df = df.dropna(subset=["NER_PRODUCT"])
# df = df.drop_duplicates(subset=["content"])

# ner_list = df["NER_PRODUCT"].str.split(",").explode().tolist()
# char_freq_dict = get_english_ngrams(ner_list)

