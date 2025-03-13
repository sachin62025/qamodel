# qamodel Library
This Library work only given dataset has columns name 'question' and another is 'answer' -> answer sholud we one word, question length can we long
## How to use 
- First install
  !pip install  qamodel
-  import Library
  from qamodel import train_model, predict
- train model
  path = 'path of dataset in csv format'
  model, vocab = train_model(path)

- predict
  question = 'question'
  answer = predict(model, vocab,question)
  print(f'Question {question} \nAnswer : {answer}')
