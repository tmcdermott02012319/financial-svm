from pickle import load

with open("vectorizer.pkl", "rb") as file:
	vectorizer = load(file)

with open("model.pkl", "rb") as file:
	model = load(file)

feature_probs = {}
for feature in vectorizer.get_feature_names_out():
	feature_probs[feature] = dict(zip(["negative", "neutral", "positive"], model.predict_proba(vectorizer.transform([feature]))[0]))

print("Most negative features:")
for (k, v) in sorted(feature_probs.items(), key = lambda x: x[1]["negative"])[:-10:-1]:
	print(f"{k}: {v["negative"]}")
print("\nMost neutral features:")
for (k, v) in sorted(feature_probs.items(), key = lambda x: x[1]["neutral"])[:-10:-1]:
	print(f"{k}: {v["neutral"]}")
print("\nMost positive features:")
for (k, v) in sorted(feature_probs.items(), key = lambda x: x[1]["positive"])[:-10:-1]:
	print(f"{k}: {v["positive"]}")
print()

while True:
	print(dict(zip(["negative", "neutral", "positive"], model.predict_proba(vectorizer.transform([input("Enter a sentence to evaluate: ")]))[0])))