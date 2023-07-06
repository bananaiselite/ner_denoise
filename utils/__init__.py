import re
from collections import defaultdict
import numpy as np

def clean_text(text, patterns):
    """
    清理文本中的模式匹配項
    text - 要清理的文本字符串
    patterns - 包含要替換的模式的列表
    return: 清理後的文本字符串
    """
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    return text

def run_clean(text):
    patterns = [
        r'[(19)|(20)]+\d{2}', 
        r'^\d+(ml|ghz|kg)+$', 
        r'^[-+]?[0-9]*\.?[0-9]?$', 
        r'^NT\d+$', 
        r'\d+\.?\d?(ml)+', 
        r'^ml$'
    ]
    return clean_text(text, patterns)

def is_number(s):
    """
    檢查字串是否為純數字
    """
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def get_ngram_dict(ner_lst, min_window=1, max_window=15, split_char=None):
    """
    創建n-gram字典，統計字符出現頻率
    
    prd_lst - 包含要處理的字符串的列表
    min_window - 最小窗口大小（預設為1）
    max_window - 最大窗口大小（預設為15）
    split_char - 分隔字符（預設為None）
    
    output: 包含字符頻率的字典
    """
    assert min_window >= 1, "minus windows size要大於1"
    
    char_freq_dict = defaultdict(int)
    
    for ner in ner_lst:
        if split_char:
            ner = ner.split(split_char)
        
        if len(ner) > max_window:
            
            for win in range(min_window, max_window+1):
                for i in range(len(ner)-win+1):
                    char = ner[i:i+win]
                    
                    if split_char:
                        char = split_char.join(char)

                    char_freq_dict[char] += 1
        else:
            for win in range(min_window, len(ner)+1):
                for i in range(len(ner)-win+1):
                    char = ner[i:i+win]

                    if split_char:
                        char = split_char.join(char)
                    char_freq_dict[char] += 1
                
    return char_freq_dict

def char_compare(candidate_char, char_freq_dict):
    """
    以詞頻選出最後留下的NER結果
    """
    max_score= 0.0
    final_char = ''
    for char in candidate_char:
        if char_freq_dict[char] > max_score:
            max_score = char_freq_dict[char]
            final_char =char
            
    return final_char, max_score

def is_sentence(text, pattern):
    # 檢查文字不為空，且長度大於 1，且不全為數字，才進行匹配
    if text and len(text) > 1 and not text.isnumeric():
        return bool(re.match(pattern, text))
    else:
        return False
    
def is_english(text):
    """
    判斷給定的文字是否為純英文
    """
    pattern = "^[^\u4e00-\u9fa5]*[A-Za-z0-9\s]+[^\u4e00-\u9fa5]*$"
    return is_sentence(text, pattern)

def is_chinese(text):
    """
    判斷給定的文字是否為純中文
    """
    pattern = r'^[\u4e00-\u9fa5-_0.-9\s]+$'
    return is_sentence(text, pattern)


def replace_multiple_spaces(string):
    # print(string)
    # new_string = re.sub(r'\s+', ' ', string)
    # return new_string

    # 使用正則表達式將連續多個空白替換為單個空白
    if isinstance(string, str):
        
        new_string = re.sub(r'\s+', ' ', string)
        return new_string
    
    else:
        return np.nan