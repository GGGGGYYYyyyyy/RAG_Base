import jieba
import re
import requests

class TextPreprocessor:
    def __init__(self, stopwords_path=None):
        """
        初始化文本预处理器，可选加载停用词表。

        :param stopwords_path: 停用词表的文件路径，如果不提供，则停用词功能禁用。
        """
        self.stopwords = set()
        if stopwords_path:
            try:
                self.stopwords = self._load_stopwords(stopwords_path)
            except Exception as e:
                print(f"加载停用词表失败: {e}")

    def _load_stopwords(self, filepath):
        """
        加载停用词表。

        :param filepath: 停用词表路径。
        :return: 停用词集合。
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())

    def preprocess_text(self, text):
        """
        对单个文本进行预处理。

        :param text: 输入的原始文本。
        :return: 预处理后的文本。
        """
        if not text or not isinstance(text, str):
            return ""
            
        # 转为小写
        text = text.lower()
        # 全角转半角
        text = self._full_to_half(text)
        # 替换数字为 <NUM>
        text = re.sub(r'\d+', '<NUM>', text)
        # 去除特殊符号，保留中文、英文、数字和常见符号
        text = re.sub(r'[^\w\s\u4e00-\u9fa5<NUM>]', '', text)
        return text

    def _full_to_half(self, s):
        """
        全角字符转半角字符。

        :param s: 输入的文本。
        :return: 转换后的文本。
        """
        return ''.join(
            chr(ord(char) - 65248) if 65281 <= ord(char) <= 65374 else char
            for char in s
        )

    def remove_stopwords(self, tokens):
        """
        移除分词结果中的停用词。

        :param tokens: 分词后的列表。
        :return: 移除停用词后的列表。
        """
        if not self.stopwords:
            return tokens
        return [token for token in tokens if token not in self.stopwords]

    def preprocess_corpus(self, corpus):
        """
        对语料进行预处理和分词。

        :param corpus: 文本语料列表。
        :return: 处理后的语料分词结果列表。
        """
        processed_corpus = []
        for text in corpus:
            # 预处理文本
            clean_text = self.preprocess_text(text)
            # 分词
            tokenized_text = list(jieba.cut(clean_text))
            # 去除停用词
            filtered_tokens = self.remove_stopwords(tokenized_text)
            processed_corpus.append(filtered_tokens)
        return processed_corpus
