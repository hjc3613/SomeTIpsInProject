import re
import json
from tqdm import tqdm

def clean_text(text):
    def special2n(string):
        # string = string.replace(r"\n", "")
        # return re.sub("[ |\t|\r|\n|\\\|\u0004]", "_", string)
        return re.sub(r'\s', '_', string)

    def strQ2B(ustr):
        "全角转半角"
        rstr = ""
        for uchar in ustr:
            inside_code = ord(uchar)
            # 全角空格直接转换
            if inside_code == 12288:
                inside_code = 32
            # 全角字符（除空格）根据关系转化
            elif (inside_code >= 65281 and inside_code <= 65374):
                inside_code -= 65248

            rstr += chr(inside_code)
        return rstr

    return strQ2B(special2n(text)).lower()


def get_dot_index(text, max_split_len):
    flag = 0
    text_ = text
    dot_index_list = []

    while (len(text_) > max_split_len):
        text_ = text_[:max_split_len]
        index_list = []
        for match in re.finditer(r"[,，;；。]", text_):
            index_list.append(match.span()[0])

        index_list.sort()
        if len(index_list) > 0:
            last_dot = index_list.pop()
        else:
            last_dot = len(text_)
        dot_index_list.append(last_dot + flag)
        text_ = text[(last_dot + flag):]
        flag += last_dot
    # 全句都没有标点符号，或者最后一个标点符号不在句末，把结尾处索引也放进来
    if not dot_index_list or dot_index_list[-1] < len(text):
        dot_index_list.append(len(text))
    return dot_index_list

def assign_label(y, start, end, label):
    assert end - start > 0, "索引不合理！！！"
    assert y[start:end] == ['O']*(end-start), '标签冲突！！！'
    if end - start > 1:
        labels = ['B-'+label] + ['I-'+label]*(end-start-2) + ['E-'+label]
    else:
        labels = ['S-'+label]
    y[start:end] = labels
    return y

def get_short_text_label(text, shorter_spans, entities, key_start='start', key_end='end', key_label='label'):
    X = []
    y = []
    for span_s, span_e in shorter_spans:
        text_ = text[span_s:span_e]
        X_ = list(text_)
        y_ = ['O']*len(X_)
        for entity in entities:
            start = entity[key_start]
            end = entity[key_end]
            label = entity[key_label]
            if start < span_s or end > span_e:
                continue
            # 实体在当前短句中的位置
            offset_s = start - span_s
            offset_e = end - span_s
            y_ = assign_label(y_, offset_s, offset_e, label)
        X.append(X_)
        y.append(y_)
    return X, y

def get_short_spans(long_txt, max_split_len, level=0, offset=0, debug=False):
    punc_lst = [r'(;|；|。|:|：|$)', r'(,|，|$)', r'(、|$)']
    
    punc = punc_lst[level]
    
    spliter = re.compile(r"(.+?)"+punc)
    # short_lst = list(spliter.finditer(long_txt))
    result = []
    for span in spliter.finditer(long_txt):
        cur_start, cur_end = span.span()
        # 当前span长度不超过max_split_len，尽量往上一个片段拼接，无法拼接时，作为新的span
        if cur_end - cur_start < max_split_len:
            if len(result) > 0 and cur_end - result[-1][0] < max_split_len:
                result[-1] = (result[-1][0], cur_end)
            else:
                result.append((cur_start, cur_end))
        else:
            # 如果当前span长度依然很长，继续深度切分
            if level > len(punc_lst)-1:
                sub_spans = [(i, i+max_split_len) for i in range(cur_start, cur_end, max_split_len)]
                result.extend(sub_spans)
            else:
                sub_spans = get_short_spans(long_txt[cur_start:cur_end], max_split_len, level=level+1, offset=cur_start)
                result.extend(sub_spans)
    
    def check_result(span):
        '''
        检查切分结果是否合理，若span长度 < max_split_len，则应该以标点符号为结尾。否则，其长度应该==max_split_len，最后一个除外
        '''
        s, e = span


    result = [(s+offset, e+offset) for s,e in result]
    if debug and level==0:
        sub_text = [long_txt[s:e] for s,e in result]
        assert ''.join(sub_text) == long_txt, "切分有未知bug"
    return result

def split_too_long_sent(text: str, entities: list, max_split_len=256, key_start='start', key_end='end', key_label='label'):
    '''
    text:
    entities: [{start:5, end:10, label:person, ent:小明}, ...]
    '''
    text = clean_text(text)
    X = []
    y = []
    entity_data = []
    if len(text) < max_split_len:
        X_ = list(text)
        y_ = ['O'] * len(X_)
        for entity in entities:
            start_pos = entity[key_start]
            end_pos = entity[key_end]
            label = entity[key_label]
            assign_label(y_, start_pos, end_pos, label)
        X.append(X_)
        y.append(y_)

    else:
        shorter_spans = get_short_spans(text, max_split_len, debug=True)
        X_list, y_list, entity_data_ = get_short_text_label(text, shorter_spans, entities, key_start=key_start, key_end=key_end, key_label=key_label)
        X.extend(X_list)
        y.extend(y_list)
        entity_data.extend(entity_data_)


def preprocess_train_dataset(path, text_col='text', entities_col='entities', key_start='start', key_end='end', key_label='label'):
    if path.endswith('.jsonl'):
        with open(path, encoding='utf8') as f:
            data = [json.loads(i.strip()) for i in f.readlines()]
    elif path.endswith('.json'):
        with open(path, encoding='utf8') as f:
            data = json.load(f)
    else:
        raise Exception('unrecognized file format: 请提供json或jsonl格式的文件')

    for idx, example in tqdm(enumerate(data), total=len(data), desc='截断长句子'):
        if idx == 108:
            print('debug')
        assert text_col in example and entities_col in example, '请提供正确的文本列名和实体列名'
        text = example[text_col]
        entities = example[entities_col]
        split_too_long_sent(text, entities, max_split_len=128,key_start=key_start,
                            key_end=key_end, key_label=key_label)


if __name__ == '__main__':
    # split_too_long_sent('')
    preprocess_train_dataset('hjc_检验检查关系抽取-第二批-136_135_20221027172213.jsonl',
                             entities_col='markedList', text_col='text', key_start='startOffset', key_end='endOffset', key_label='tagName')
