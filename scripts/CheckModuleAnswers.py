global db1
global db2
global db0
global compliments
global encur

compliments = ['Good job!', 
               'Well done!']

encour = ['Better luck next time.', 
        'Not completely correct.', 
        'Please try again.', 
        'You are so close, try again.']

db1 = {'1':'O55222',
       '2':'YES!'
      }
db2 = {'1':'DATASCIENCE' , 
       '3':'Please_fill_out'
      }
db0 = {'1':'O55222' 
      }


import numpy as np
def give_compliment():
    print(np.random.choice(compliments, 1)[0])
    
def encourage():
    print(np.random.choice(encour, 1)[0])


def check_answer_module1(question, answer):
    """Checks the question and answer pair against a the correct answer database. 
    No mismatches / typos are allowed.
    
    question : Question number.
    answer : Your answer"""

    if str(question) in db1:
        correct = db1[str(question)]
        if correct == str(answer):
            give_compliment()
            print('your answer was : {}'.format(answer))
        else:
            encourage()
            print('your answer was : {}'.format(answer))
    else:
        print("""It doesn't seem like you typed a valid question.""")
        
def check_answer_module2(question, answer):
    """Checks the question and answer pair against a the correct answer database. 
    No mismatches / typos are allowed.
    
    question : Question number.
    answer : Your answer"""

    if str(question) in db2:
        correct = db2[str(question)]
        if correct == str(answer):
            give_compliment()
            print('your answer was : {}'.format(answer))
        else:
            encourage()
            print('your answer was : {}'.format(answer))
    else:
        print("""It doesn't seem like you typed a valid question.""")
        
def check_answer_module0(question, answer):
    """Checks the question and answer pair against a the correct answer database. 
    No mismatches / typos are allowed.
    
    question : Question number.
    answer : Your answer"""
    if str(question) in db0:
        correct = db0[str(question)]
        if correct == str(answer):
            give_compliment()
            print('your answer was : {}'.format(answer))
        else:
            encourage()
            print('your answer was : {}'.format(answer))
    else:
        print("""It doesn't seem like you typed a valid question.""")