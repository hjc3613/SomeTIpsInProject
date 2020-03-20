from collections import defaultdict, abc


class _defaultdict(defaultdict):
    def __add__(self, other):
        return other


def NestDict():
    return _defaultdict(NestDict)


def insertNewValue(old_dict, new_value):
    '''
    给一个dictionary插入一些新的值，当新插入的值是nested结构时，用到此函数
    :param old_dict: {'a':1, 'b':{'c':2}}
    :param new_value: {'b':{'d':3}}
    :return:{'a': 1, 'b': {'c': 2, 'd': 3}}
    '''
    for k, v in new_value.items():
        if isinstance(v, abc.Mapping):
            old_dict[k] = insertNewValue(old_dict.get(k, {}), v)
        else:
            old_dict[k] = v
    return old_dict


if __name__ == '__main__':
    from pprint import pprint
    merge_result = {'a': 1, 'b': {'c': 2, 'e':{'f':4}}}
    assistant_respiration = {'b': {'e': 3, 'f':{'g':5}}}
    new_value = NestDict()
    path = 'de_value_quantization.interventions.interventions_details.assisted_breathing'
    path = path.split('.')
    new_value_path = ''.join(list("['" + i + "']" for i in path))
    exec(f'new_value{new_value_path} = assistant_respiration')
    merge_result = insertNewValue(merge_result, new_value)
    pprint(merge_result)
