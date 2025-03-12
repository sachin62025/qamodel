from qamodel import train_model, predict

model, vocab = train_model("C:/Users/sachi/python/9-Deep-Learning/Pytorch/100_Unique_QA_Dataset.csv")
print(predict(model, vocab, "What is the capital of France?"))
# print(predict(model, vocab, "What is the capital of France?"))