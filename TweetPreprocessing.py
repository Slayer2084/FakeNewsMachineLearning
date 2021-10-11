import re
import pandas as pd
import emoji
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from edm import report
from emoticons import EMOTICONS
from collections import Counter


class Error(Exception):
    """"Klasse für andere Fehler"""
    pass


class NullsInDataframe(Error):
    """"Fehler Meldung für nulls im df"""
    pass


class NansInDataframe(Error):
    """"Fehler Meldung für nans im df"""
    pass


class Preprocessing:
    def __init__(self, dframe):
        self.df = dframe
        self.shape = self.df.shape

        if self.df["tweet"].isnull().sum() > 0:
            raise NullsInDataframe
        if self.df["tweet"].isna().sum() > 0:
            raise NansInDataframe

    def remove_names(self, chain=True):
        def rm(text):
            if "@" in text:
                text = re.sub('@.*? ', '@[name] ', text)
                if "@" in text.split()[-1]:
                    lastword = text.split()[-1]
                    text = text.replace(lastword, "@[name] ")
                if "RT" in text.split()[0]:
                    text = re.sub('RT.*? ', '', text)
                return text
            else:
                return text

        self.df['tweet'] = self.df['tweet'].copy().apply(rm)
        if chain:
            return Preprocessing(self.df.copy())
        return self.df.copy()

    def remove_chars(self, chain=True):
        self.df = self.df.copy().replace(to_replace=["&amp;", "&gt;", "&lt;", r"\n", r"\r", r"\n\r", r"\r\n",
                                                     r"\r\n\r\n", "​", "�"],
                                         value=["&", ">", "<", "", "", "", "", "", "", ""], regex=True)
        if chain:
            return Preprocessing(self.df.copy())
        return self.df.copy()

    def remove_spaces(self, chain=True):
        self.df['tweet'] = self.df['tweet'].copy().apply(lambda text: re.sub(' +', ' ', text))
        if chain:
            return Preprocessing(self.df.copy())
        return self.df.copy()

    def lemmatize(self, chain=True):
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        lemmatizer = WordNetLemmatizer()
        wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

        def lemmatize_words(text):
            pos_tagged_text = nltk.pos_tag(text.split())
            return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN))
                             for word, pos in pos_tagged_text])

        self.df["tweet"] = self.df["tweet"].copy().apply(lambda text: lemmatize_words(text))
        if chain:
            return Preprocessing(self.df.copy())
        return self.df.copy()

    def convert_emoticons(self, chain=True):
        """Funktioniert zur Zeit nicht optimal, da auch nicht Emoticons mit der gleichen Zeichenfolge erkannt und
        umgewandelt werden"""

        def cnvrt_emtcns(text):
            for emot in EMOTICONS:
                text = re.sub(u'(' + emot + ')', "_".join(EMOTICONS[emot].replace(",", "").split()), text)
            return text

        self.df["tweet"] = self.df["tweet"].copy().apply(lambda text: cnvrt_emtcns(text))
        if chain:
            return Preprocessing(self.df.copy())
        return self.df.copy()

    def convert_emoji(self, chain=True):
        def convert_emojis(text):
            return emoji.demojize(text, delimiters=("", " "))

        self.df["tweet"] = self.df["tweet"].copy().apply(lambda text: convert_emojis(text))
        if chain:
            return Preprocessing(self.df.copy())
        return self.df.copy()

    def add_pol_subj_to_df(self, chain=True):
        self.df['sentiment'] = self.df['tweet'].apply(lambda tweet: TextBlob(tweet).sentiment)
        if chain:
            return Preprocessing(self.df.copy())
        return self.df.copy()

    def get_report(self):
        sents = self.df["tweet"].values
        labels = self.df["label"].values
        print(report.get_difficulty_report(sents, labels))

    def remove_rare_words(self, chain=True, n_rare_words=10):
        cnt = Counter()
        for text in self.df["tweet"].values:
            for word in text.split():
                cnt[word] += 1
        rarewords = set([w for (w, wc) in cnt.most_common()[:-n_rare_words - 1:-1]])

        def remove_rarewords(txt):
            return " ".join([_word for _word in str(txt).split() if _word not in rarewords])

        self.df["tweet"] = self.df["tweet"].copy().apply(lambda texts: remove_rarewords(texts))
        if chain:
            return Preprocessing(self.df.copy())
        return self.df.copy()


if __name__ == "__main__":
    df = pd.read_csv("2Cleaned_Fake_News_Dataset.csv", index_col='index', sep=';')
    preprocessing = Preprocessing(df)
    preprocessing.get_report()
