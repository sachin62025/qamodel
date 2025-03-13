from qamodel import train_model, predict
path = r"C:\Users\sachi\python\5-practice-ML\more\cl\cleaned_merged_dataset.csv"
model, vocab = train_model(path)
print(predict(model, vocab, "Which country has the pyramids of Giza?"))