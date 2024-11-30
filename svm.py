from sklearn.model_selection import train_test_split
from imblearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from pickle import load, dump

with open("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt", "r", encoding = "latin1") as file: # no idea what encoding it actually is but utf-8 doesnt work
	data = file.read()

data = [line.split("@") for line in data.split("\n") if line]
x = [line[0] for line in data]
y = [line[1] for line in data]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) # train = 80%, test = 20%

try:
	with open("model.pkl", "rb") as file:
		model = load(file)
except FileNotFoundError:
	pipeline = make_pipeline(
		TfidfVectorizer(),
		MaxAbsScaler(),
		SMOTE(),
		SVC(),
	)
	params = {
		"tfidfvectorizer__ngram_range": [(1, 1), (1, 2)],
		"tfidfvectorizer__min_df": [0.001, 0.01, 0.1],
		"tfidfvectorizer__max_df": [0.5, 0.75, 1.0],
		"tfidfvectorizer__sublinear_tf": [True, False],
		"smote__k_neighbors": [1, 2, 5, 10],
		"smote__random_state": [0],
		"svc__C": [0.1, 1, 10],
		"svc__kernel": ["linear", "poly", "rbf"],
		"svc__gamma": ["scale", "auto"],
		"svc__class_weight": ["balanced"],
		"svc__random_state": [0],
	}
	grid = GridSearchCV(pipeline, params, scoring="balanced_accuracy", n_jobs=-1, cv=3, verbose=3) # due to cross validation the validation set goes unused
	grid.fit(x_train, y_train)
	model = grid.best_estimator_
	print("Best params:", grid.best_params_)
	with open("model.pkl", "wb") as file:
		dump(model, file)

y_pred = model.predict(x_test)
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1:", f1_score(y_test, y_pred, average="weighted"))