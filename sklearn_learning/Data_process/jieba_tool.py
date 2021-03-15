import jieba
"""
Author: Siliang Liu
15/08/2020
Reference itheima.com
"""


def separate_word(text):
    generator = jieba.cut(text)  # 分词生成器
    text_array = list(generator)  # 转换为数组
    final_text = " ".join(text_array)
    return final_text

