import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ========== 0. 自动切换到脚本所在目录 ==========
os.chdir(os.path.dirname(__file__))
print("Current working directory:", os.getcwd())

# ========== 1. 读取数据 ==========
cols = [
    "json_id", "label", "statement", "subject", "speaker", "speaker_job_title",
    "state_info", "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]

train = pd.read_csv("train.tsv", sep="\t", header=None, names=cols)
valid = pd.read_csv("valid.tsv", sep="\t", header=None, names=cols)
test  = pd.read_csv("test.tsv",  sep="\t", header=None, names=cols)

print("Train:", train.shape)
print("Valid:", valid.shape)
print("Test :", test.shape)
print(train["label"].value_counts())

# ========== 2. 缺失值与重复值处理 ==========
train = train.dropna(subset=["statement", "label"])
train = train.drop_duplicates(subset=["statement", "speaker", "context"])

print("\nMissing value ratio (top 5):")
print(train.isnull().mean().sort_values(ascending=False).head())

# ========== 3. 文本清洗函数 ==========
# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# from nltk.stem import WordNetLemmatizer
# lemm = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9%?!$ ]", " ", text)
    # 临时关闭词形还原以避免 NLTK 下载卡死
    # text = ' '.join(lemm.lemmatize(w) for w in text.split())
    return text.strip()

train["clean_statement"] = train["statement"].apply(clean_text)
valid["clean_statement"] = valid["statement"].apply(clean_text)
test["clean_statement"]  = test["statement"].apply(clean_text)

# ========== 4. 可视化：语句长度分布 & 标签分布 ==========
train["len"] = train["clean_statement"].apply(lambda x: len(x.split()))

plt.figure(figsize=(6,4))
sns.histplot(train["len"], bins=30, color="steelblue")
plt.title("Distribution of Statement Lengths")
plt.xlabel("Number of tokens")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("len_dist.png")
plt.close()

plt.figure(figsize=(6,4))
sns.countplot(y=train["label"], order=train["label"].value_counts().index, palette="viridis")
plt.title("Label Distribution in Training Set")
plt.xlabel("Count")
plt.ylabel("Label")
plt.tight_layout()
plt.savefig("label_dist.png")
plt.close()

print("\n✅ Saved visualization: len_dist.png, label_dist.png")

# ========== 5. TF-IDF 特征提取 ==========
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train = tfidf.fit_transform(train["clean_statement"])
X_valid = tfidf.transform(valid["clean_statement"])
X_test  = tfidf.transform(test["clean_statement"])

print("TF-IDF matrix shape:", X_train.shape)
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

# ========== 6. 保存清洗后的数据 ==========
train.to_csv("train_clean.csv", index=False)
valid.to_csv("valid_clean.csv", index=False)
test.to_csv("test_clean.csv", index=False)

print("\n✅ Saved cleaned datasets:")
print("   train_clean.csv, valid_clean.csv, test_clean.csv")
print("✅ Saved TF-IDF vectorizer: tfidf_vectorizer.pkl")

print("\n🎯 All preprocessing steps completed successfully!")
print("   Clean data ready for model 🚀")