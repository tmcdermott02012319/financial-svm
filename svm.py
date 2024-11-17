from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def load_dataset():
	with open("FinancialPhraseBank-v1.0/Sentences_50Agree.txt", "r", encoding = "latin1") as file:
		data = file.read()
	data = [line.split("@") for line in data.split("\n") if line]
	x = [line[0] for line in data]
	y = [line[1] for line in data]
	# train = 60%, test = 20%, val = 20%
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)
	return x_train, x_test, x_val, y_train, y_test, y_val

def extract_features(x_train):
	vectorizer = TfidfVectorizer()
	x_train = vectorizer.fit_transform(x_train)
	return x_train, vectorizer

def train_model(x_train, y_train):
	model = LinearSVC()
	return model.fit(x_train, y_train)

def evaluate_model(model, x_test, y_test):
	y_pred = model.predict(x_test)
	print(classification_report(y_test, y_pred)) # TODO: balanced accuracy

if __name__ == "__main__":
	x_train, x_test, x_val, y_train, y_test, y_val = load_dataset()
	x_train, vectorizer = extract_features(x_train) # TODO: over/undersampling
	x_test = vectorizer.transform(x_test)
	x_val = vectorizer.transform(x_val)
	model = train_model(x_train, y_train)
	evaluate_model(model, x_test, y_test)