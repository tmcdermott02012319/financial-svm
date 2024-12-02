from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE


def load_dataset():
	with open("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt", "r", encoding = "latin1") as file:
		data = file.read()
	data = [line.split("@") for line in data.split("\n") if line]
	x = [line[0] for line in data]
	y = [line[1] for line in data]
	# train = 60%, test = 20%, val = 20%
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, stratify=y_train)
	return x_train, x_test, x_val, y_train, y_test, y_val


def extract_features(x_train):	
	vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.5, sublinear_tf=True, stop_words='english')
	x_train = vectorizer.fit_transform(x_train)
	return x_train, vectorizer

def train_model(x_train, y_train, x_val, y_val):
	model = SVC(class_weight="balanced", probability=True)
	params = {
		"C": [0.1, 1, 10],
		"kernel": ["linear", "poly", "rbf", "sigmoid"],
		"gamma": [0.1, 1, "scale", "auto"]
	}
	optimizer = GridSearchCV(model, params, scoring="balanced_accuracy", n_jobs=-1)
	optimizer.fit(x_val, y_val)
	return optimizer.best_estimator_.fit(x_train, y_train)

def evaluate_model(model, x_test, y_test):
	y_pred = model.predict(x_test)
	print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
	print("Precision:", precision_score(y_test, y_pred, average="weighted"))
	print("Recall:", recall_score(y_test, y_pred, average="weighted"))
	print("F1:", f1_score(y_test, y_pred, average="weighted"))

if __name__ == "__main__":
	x_train, x_test, x_val, y_train, y_test, y_val = load_dataset()
	x_train, vectorizer = extract_features(x_train)
	x_test = vectorizer.transform(x_test)
	x_val = vectorizer.transform(x_val)
	x_res, y_res = SMOTE().fit_resample(x_train, y_train)
	model = train_model(x_res, y_res, x_val, y_val)
	evaluate_model(model, x_test, y_test)