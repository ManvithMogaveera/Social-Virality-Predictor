import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import joblib
 
print("🔄 Loading source datasets...")
df_comments = pd.read_csv('comments.csv')
df_posts    = pd.read_csv('posts.csv')
 

print("🔗 Merging post metrics onto comments...")
df_comments['extracted_post_id'] = df_comments['permalink'].str.extract(r'/comments/([a-z0-9]{6})/')
df_posts_clean = df_posts.rename(columns={'score': 'post_score'})
 
df_posts_clean['post_title_len']  = df_posts_clean['title'].str.len().fillna(0)
df_posts_clean['post_body_len']   = df_posts_clean['selftext'].str.len().fillna(0)
df_posts_clean['post_is_question']= df_posts_clean['title'].str.contains(r'\?', regex=True).astype(int)
 
post_cols = ['id', 'title', 'post_score', 'post_title_len', 'post_body_len', 'post_is_question']
df = pd.merge(df_comments, df_posts_clean[post_cols],
              left_on='extracted_post_id', right_on='id', how='left')
 

df['sentiment']        = df['sentiment'].fillna(0.0)
df['body']             = df['body'].fillna('')
df['extracted_post_id']= df['extracted_post_id'].fillna('unknown')
df['title']            = df['title'].fillna('No Parent Title')
 

print("🧹 Filtering and log-transforming target...")
df_cleaned = df[df['score'] >= 0].copy()
df_cleaned['logged_score_fixed'] = np.log1p(df_cleaned['score'])
 

df_cleaned['word_count']          = df_cleaned['body'].apply(lambda x: len(str(x).split()))
df_cleaned['char_count']          = df_cleaned['body'].str.len()
df_cleaned['is_short_comment']    = (df_cleaned['word_count'] < 5).astype(int)
df_cleaned['created_utc']         = pd.to_datetime(df_cleaned['created_utc'], unit='s')
df_cleaned['hour']                = df_cleaned['created_utc'].dt.hour
df_cleaned['day_of_week']         = df_cleaned['created_utc'].dt.day_of_week
df_cleaned['sentiment_intensity'] = df_cleaned['sentiment'].abs()
df_cleaned['log_post_score']      = np.log1p(df_cleaned['post_score'].clip(lower=0))
df_cleaned['is_weekend']          = (df_cleaned['day_of_week'] >= 5).astype(int)
df_cleaned['is_peak_hour']        = df_cleaned['hour'].isin([12,13,14,15,20,21,22]).astype(int)
df_cleaned['comment_depth']       = df_cleaned['permalink'].str.count('/') - 6
 

print("🚀 Vectorising post titles with TF-IDF (bigrams, 50 features)...")
tfidf = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df_cleaned['title'])
 
viral_words_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f"title_word_{w}" for w in tfidf.get_feature_names_out()],
    index=df_cleaned.index
)
df_final = pd.concat([df_cleaned, viral_words_df], axis=1)
 

word_features = [f"title_word_{w}" for w in tfidf.get_feature_names_out()]
base_features = [
    'sentiment', 'sentiment_intensity', 'hour', 'day_of_week', 'word_count',
    'post_score', 'post_title_len', 'post_body_len', 'post_is_question',
    'char_count', 'is_short_comment', 'log_post_score', 'is_weekend',
    'is_peak_hour', 'comment_depth'
]
features = base_features + word_features
 

df_final['score_strata'] = pd.cut(
    df_final['score'], bins=[-1,1,3,10,np.inf],
    labels=['low','Medium','High','Viral']
)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
 
for train_idx, test_idx in splitter.split(df_final, df_final['score_strata']):
    strata_train = df_final.iloc[train_idx]
    strata_test  = df_final.iloc[test_idx]
 
X_train = strata_train[features]
y_train = strata_train['logged_score_fixed']
X_test  = strata_test[features]
y_test  = strata_test['logged_score_fixed']
 
print(f"\n--- Split confirmed: Train {X_train.shape} | Test {X_test.shape} ---")
print("Virality class distribution in test set (%):")
print(strata_test['score_strata'].value_counts(normalize=True).mul(100).round(2))
 

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, features)
])
 

print("\n🏋️  Training XGBoost regressor...")
xgb_model = XGBRegressor(
    tree_method='hist',
    device='cuda',          # change to 'cpu' if no GPU
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    random_state=42,
    n_jobs=-1
)
 
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])
 
full_pipeline.fit(X_train, y_train)
 

y_pred_log = full_pipeline.predict(X_test)
y_pred     = np.expm1(y_pred_log)
y_true     = np.expm1(y_test)
 
rmse = np.sqrt(mean_squared_error(y_test, full_pipeline.predict(X_test)))
r2   = r2_score(y_test, full_pipeline.predict(X_test))
 
print("\n📈 --- Pipeline Performance Metrics (log-space) ---")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")
 
importances = pd.Series(
    xgb_model.feature_importances_,
    index=features
).sort_values(ascending=False)
print("\n📊 Top 10 Feature Importances:")
print(importances.head(10))
 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load datasets

comments = pd.read_csv("comments.csv")
posts = pd.read_csv("posts.csv")

# -------------------------------

# 1. Score Distribution

# -------------------------------

plt.figure(figsize=(8,5))
sns.histplot(comments['score'], bins=100, kde=True)
plt.xlim(0, 200)
plt.title("Comment Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/score_distribution.png")

# -------------------------------

# 2. Sentiment vs Score

# -------------------------------

plt.figure(figsize=(8,5))
sns.scatterplot(
x=comments['sentiment'],
y=np.log1p(comments['score']),
alpha=0.4
)
plt.title("Sentiment vs Log(Comment Score)")
plt.xlabel("Sentiment")
plt.ylabel("Log Score")
plt.tight_layout()
plt.savefig("plots/sentiment_vs_score.png")

# -------------------------------

# 3. Virality Classes

# -------------------------------

bins = [-1,1,3,10,np.inf]
labels = ['Low','Medium','High','Viral']

comments['virality_class'] = pd.cut(
comments['score'],
bins=bins,
labels=labels
)

comments['virality_class'].value_counts().plot(
kind='bar',
figsize=(7,5)
)

plt.title("Virality Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/virality_classes.png")

# -------------------------------

# 4. Post Score vs Comment Score

# -------------------------------

merged = comments.merge(
posts[['id','score']],
left_on='id',
right_on='id',
suffixes=('_comment','_post')
)

plt.figure(figsize=(8,5))
sns.scatterplot(
x=np.log1p(merged['score_post']),
y=np.log1p(merged['score_comment']),
alpha=0.4
)

plt.title("Post Score vs Comment Score")
plt.xlabel("Log Post Score")
plt.ylabel("Log Comment Score")
plt.tight_layout()
plt.savefig("plots/post_score_vs_comment_score.png")

print("Plots saved successfully.")
    
# joblib.dump(xgb_model, 'social_virality_xgb_model.pkl')
# joblib.dump(tfidf,     'virality_tfidf_vectorizer.pkl')
# print("\n✅ Model assets saved:")
# print("   → social_virality_xgb_model.pkl")
# print("   → virality_tfidf_vectorizer.pkl")
# print("\n🎉 Run 'streamlit run app.py' to launch the dashboard.")