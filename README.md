# 🚀 Russian Male Last Name Generator

This repository contains the code and resources for training and evaluating a neural network model designed to generate Russian male last names.

---

This research project aims to train and evaluate a character-based language model designed to generate Russian male last names. 
The evaluation of the model's quality is approached through three distinct methods, each offering insights into its performance. Firstly, the model's output will be assessed using the superior model, OpenAI ChatGPT, which will judge the authenticity of the generated last names. Even though ChatGPT is considered a less reliable tool than a human evaluator, it can provide valuable information and save us time and resources otherwise spent on manual labor.
Secondly, a language-specific metric will be employed. We will focus on particular suffixes that serve as markers of Russian last names. By conducting this evaluation, we will understand the model's ability to capture linguistic nuances.
Lastly, the generated last names will be cross-referenced with Google Search results to validate their real-world existence. This method is used to ensure the relevance of generated last names for practical applications. Additionally, this research will briefly address the challenges associated with training a language model on non-Latin-based scripts, such as the Cyrillic script used in the Russian language. Our expectation is for the model to generate both the last names that really exist and the ones that closely resemble Russian surnames. We expect the model to capture common suffixes. However, due to the relatively limited dataset used for training, we anticipate that the model may not generate a substantial number of last names that can be found via search engines.
Material and Method:
Andrei Karphaty’s code was used as a template for this research project: (https://colab.research.google.com/drive/1YIfmkftLrz6MPTOO9Vwqrop2Q5llHIGK?usp=sharing#scrollTo=EA A0_oigc13X). The dataset of Russian male last names was obtained from the repository available on GitHub: https://github.com/Raven-SL/ru-pnames-list/blob/master/lists/male_surnames_rus.txt. This dataset consists of 14 651 lines, each containing a single last name. Each last name starts with a capital letter.
- We chose not to convert the letters to lowercase and instead build a vocabulary of characters that consists of 61 characters, including 33 lowercase and 28 uppercase, along with a dot. This way we ensure that the model's character vocabulary aligns with Russian orthography. While the Russian alphabet consists of 33 letters, careful considerations were made regarding capitalization rules and unique letter properties. The letter Ë often loses its dots and appears in writing as another letter - Е - due to native speakers' ability to distinguish them from the context. Three letters (ы, ъ, ь) never appear in capitalized forms at the beginning of words. The letter Й theoretically allows capitalization, but such instances are limited to words foreign to the Russian language. Our model captured the capitalization patterns present in the dataset, consistently generating last names that start with capital letters.
- In the process of initializing parameters for our neural network model, we encountered an error related to hard-coding specific values for the number of characters (in Karpathy's work number 27 was used for English). This approach limited the adaptability of our code to different languages, incl. Russian. To address this issue, we replaced the hard-coded value with the variable N_chars, which represents the total number of unique characters in the vocabulary. This modification improved the flexibility and scalability of our code.
- After generating the last names, we removed the "." at the end of each word. This preprocessing step simplified the evaluation process, for example the search for suffixes, and enhanced the clarity of evaluation procedures.
- To minimize the loss and improve the performance of our neural network model, we experimented with adjusting hyperparameters. We had a learning rate that started at 0.01 and decreased it to 0.001. We updated it to a higher one: lr = 0.1 if i < 100000 else 0.01. The model’s output got higher evaluation scores with the adjusted learning rate. However, upon closer inspection, we observed that the model generated some names that already existed in the dataset. We went back to the original setting.
   
Results:
The evaluation process of our character-based language model involved three methods aimed at assessing the model's ability to capture linguistic nuances, as well as the authenticity and real-world relevance of generated last names. Evaluating the quality of model predictions, even for a native Russian speaker, can be tricky. The Russian-speaking world is a mixture of ethnicities. Regional influences often result in lesser-known or uncommon last names, particularly in areas like the Caucasus, Tatarstan (territories with a Muslim majority within Russia), or Kazakhstan (former Soviet Union republic), where names were russified with typical Russian suffixes, altering the original stems.
Even though Russian last name structures and phonetics exhibit variability, yet certain patterns prevail. Very obvious markers are suffixes: according to Wikipedia, from 60 to 70% of Russian male last names have suffixies -ов/-ев/-ёв (-ov/-ev). Besides these, last names often end in -ин/-ын (-in/-yn), -их/ -ых (-ikh/-ykh), -ский/-цкий (-skiy/-tskiy) or -ый/-ой/-ий (-yy/-oy/-iy) (with the last ones serving as gender-specific case markers). Russian last names sometimes include patronymic elements derived from the father's first name. For example, -ович (-ovich) for males. Moreover, Russian last names display diverse origins, including German (-mann, -er), Jewish (-berg), Ukrainian (-o, -enko, -yuk, -chuk), Armenian (-yan), Georgian (-dze), Polish, etc.
Firstly, we assessed the model's performance against established linguistic patterns, particularly common suffixes found in Russian male last names. The mean proportion of last names containing one of these suffixes was calculated to be 0.91, indicating that our model managed to capture this linguistic feature. The method has two primary weaknesses: firstly, the list of suffixes, while comprehensive, was not complete. Secondly, our evaluation focused solely on one morpheme, disregarding potential inadequacies in the stem.
Then, we employed a superior model, ChatGPT, to assess the authenticity of each generated last name. While human evaluation might have been more reliable, ChatGPT's scalability and cost-effectiveness made it a preferable choice. The proportion of generated last names judged as authentic by ChatGPT is 0.69. While this number indicates some level of success, there is still room for improvement. The method has obvious limitations, a superior model is prone to mistakes and biases, and it doesn’t have such a deep understanding of cultural aspects, as a native speaker.
Finally, we examined the discrepancies between model-generated names and real-world examples. To consider a last name as existing, we set a criteria of at least 1000 search results in Google. The proportion of generated last names validated through this method was 0.32. Although the model managed to generate some last names with high occurrences in Google due to famous individuals, the overall validation rate was relatively low. This outcome was expected due to the size of our model and the amount of training data. A notable weakness of this method is that Google may not always distinguish between last names and other types of words, such as common nouns. The search is case-insensitive, which can lead to false positives in the validation process. Overall, considering the size of our language model, its performance is satisfactory. It captured the capitalization of last names and incorporated the most common suffixes used in Russian male surnames. However, further refinement of the model is necessary to improve the authenticity of generated content and its real-world applicability.



## Contents

1. **Training_the_model_2_evaluations.ipynb**: Jupyter Notebook file containing model training and two evaluation methods for the model: Google search validation and suffixes.
2. **GPT_eval_API.ipynb**: Jupyter Notebook file containing the evaluation process using the superior model, ChatGPT.
3. **predicted_last_names.csv**: CSV file containing the generated last names produced by the trained model.

---

## Dependencies

- torch: PyTorch, a popular machine learning library.
- torch.nn.functional as F: Functional API of PyTorch for neural network operations.
- matplotlib.pyplot as plt: Matplotlib library for creating figures and plots.
- re: Regular expression module for pattern matching and string manipulation.
- %matplotlib inline: IPython magic command for inline plotting in Jupyter notebooks.
- random: Python's built-in random number generation module.
- pandas as pd: Pandas library for data manipulation and analysis.
- requests: Library for making HTTP requests.
- BeautifulSoup from bs4: Beautiful Soup library for web scraping and parsing HTML/XML documents.
- OpenAI: Library or module from OpenAI, possibly for accessing their APIs or services.
- os: Python's built-in module for interacting with the operating system.
- numpy as np: NumPy library for numerical computing.
- tqdm: Library for creating progress bars in Python.

---

