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

def assign_label(y, start, end, label, right=1, ent=None, text=None):
    '''
    right=1,实体索引右闭，否则右开
    '''
    # 子片段是通过**左闭右开**的方式获取的，但是当right==1时，实体是通过**左闭右闭**的方式获取的，因此当ent正好处于句尾时，不对其进行右扩；当其位于句中时，需要右扩一位变为左闭右开
    # 若right=1，实体右边界为闭区间，应改为开区间
    end = end + right
    assert end - start > 0, f"索引不合理！！！start={start}, end={end}"
    assert y[start:end] == ['O']*(end-start), f'标签冲突！！！y[{start}:{end}]={y[start:end]}, start={start}, end={end}, len(y)={len(y)}'
    if ent and text:
        if text[start:end].lower().replace('(', '（').replace(')', '）').replace('，', ',') != ent.lower().replace('(', '（').replace(')', '）').replace('，', ',') :
            print(f'{text[start:end]:10}\t{ent:10}')
    if end - start > 1:
        labels = ['B-'+label] + ['I-'+label]*(end-start-2) + ['E-'+label]
    else:
        labels = ['S-'+label]
    y[start:end] = labels
    return y

def get_short_text_label(text, shorter_spans, entities, key_start='start', key_end='end', key_label='label', right=1):
    '''
    right = 1 代表实体边界为右闭区间，若实体为左闭右开，需将right=0
    '''
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
            ent = entity.get('ent')
            # 根据right是否为1，确定实体右边界是否为闭区间，如果是闭区间，需要往右扩1，通过end+right实现
            if start < span_s or end + right > span_e:
                continue
            # 实体在当前片段中的位置
            offset_s = start - span_s
            offset_e = end - span_s
            y_ = assign_label(y_, offset_s, offset_e, label, ent=ent, text=text_, right=1)
        X.append(X_)
        y.append(y_)
    return X, y

def get_short_spans(long_txt, max_split_len, level=0, offset=0):
    punc_lst = [r'(;|；|。|:|：|$)', r'(,|，|$)', r'(、|$)']
    
    punc = punc_lst[level]
    
    spliter = re.compile(r"(.+?)"+punc)
    # short_lst = list(spliter.finditer(long_txt))
    result = []
    for span in spliter.finditer(long_txt):
        cur_start, cur_end = span.span()
        # 当前span长度不超过max_split_len，尽量往上一个片段拼接，无法拼接时，作为新的span
        if cur_end - cur_start < max_split_len:
            if len(result) > 0 and cur_end - result[-1][0] < max_split_len and level>0:
                result[-1] = (result[-1][0], cur_end)
            else:
                result.append((cur_start, cur_end))
        else:
            # 如果当前span长度依然很长，继续深度切分
            if level + 1 > len(punc_lst)-1:
                sub_spans = [(i, min(i+max_split_len, len(long_txt))) for i in range(cur_start, cur_end, max_split_len)]
                result.extend(sub_spans)
            else:
                sub_spans = get_short_spans(long_txt[cur_start:cur_end], max_split_len, level=level+1, offset=cur_start)
                sub_text = [long_txt[s:e] for s,e in sub_spans]
                assert ''.join(sub_text) == long_txt[cur_start:cur_end], "切分有未知bug"
                if result:
                    assert sub_spans[0][0] == result[-1][1]
                result.extend(sub_spans)
    
    def check_result(span):
        '''
        检查切分结果是否合理，若span长度 < max_split_len，则应该以标点符号为结尾。否则，其长度应该==max_split_len，最后一个除外
        '''
        s, e = span


    result = [(s+offset, e+offset) for s,e in result]
    if level==0:
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
            ent = entity.get('ent')
            assign_label(y_, start_pos, end_pos, label, right=1, ent=ent, text=text)
        X.append(X_)
        y.append(y_)

    else:
        shorter_spans = get_short_spans(text, max_split_len)
        X_list, y_list = get_short_text_label(text, shorter_spans, entities, key_start=key_start, key_end=key_end, key_label=key_label)
        X.extend(X_list)
        y.extend(y_list)
    return X, y

def preprocess_train_dataset(path, text_col='text', entities_col='entities', key_start='start', key_end='end', key_label='label'):
    if path.endswith('.jsonl'):
        output_path = path.replace('.jsonl', '_fixed.jsonl')
        with open(path, encoding='utf8') as f:
            data = [json.loads(i.strip()) for i in f.readlines()]
    elif path.endswith('.json'):
        output_path = path.replace('.json', '_fixed.jsonl')
        with open(path, encoding='utf8') as f:
            data = json.load(f)
    else:
        raise Exception('unrecognized file format: 请提供json或jsonl格式的文件')

    result = []
    for idx, example in tqdm(enumerate(data), total=len(data), desc='截断长句子'):
        if idx==7816:
            print()
        assert text_col in example and entities_col in example, '请提供正确的文本列名和实体列名'
        text = example[text_col]
        entities = example[entities_col]
        X_list, y_list = split_too_long_sent(text, entities, max_split_len=128,key_start=key_start,
                            key_end=key_end, key_label=key_label)
        result.extend([{'text':x, 'labels':y, 'idx':f'{idx}_{offset}', 'orig_text':text} for offset, (x,y) in enumerate(zip(X_list, y_list))])
    with open(output_path, mode='w') as f:
        f.write('\n'.join([json.dumps(i, ensure_ascii=False) for i in result]))

if __name__ == '__main__':
    # split_too_long_sent('')
    preprocess_train_dataset('/nlp/hujunchao/medbert-operation/data_disease/source/generated_train_dataset/val_dataset.json',
                             entities_col='result', text_col='text', key_start='startOffset', key_end='endOffset', key_label='tagName')
