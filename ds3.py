#Aditi Ajay Marar-DS Q

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def txt(df):
    df["Review Title"] = df["Review Title"].fillna("")
    df["Review Text"] = df["Review Text"].fillna("")
    return (df["Review Title"] + " " + df["Review Text"]).str.strip()

def cln(t):
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"\d+", "", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r"\s+", " ", t).strip()
    return t

tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")
x = txt(tr).str.lower()
xt = txt(te).str.lower()
y = tr["Star Rating"]


x = x.apply(cln)
xt = xt.apply(cln)

xtr, xvl, ytr, yvl = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

vec = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=30000,
    min_df=3
)

xtr = vec.fit_transform(xtr)
xvl = vec.transform(xvl)
xtv = vec.transform(xt)

clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

clf.fit(xtr, ytr)

p = clf.predict(xvl)

print("accuracy:", accuracy_score(yvl, p))
print("precision:", precision_score(yvl, p, average="weighted"))
print("recall:", recall_score(yvl, p, average="weighted"))
print("f1:", f1_score(yvl, p, average="weighted"))

vec = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=30000,
    min_df=3
)

xf = vec.fit_transform(x)
xtf = vec.transform(xt)

clf.fit(xf, y)

out = pd.DataFrame({
    "id": te["id"],
    "Star Rating": clf.predict(xtf)
})

out.to_csv("submission.csv", index=False)
