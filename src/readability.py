# Readability features from Chin et. al.


import textstat

def coleman_liau(text):
    return textstat.coleman_liau_index(text)

def dale_chall(text):
    return textstat.dale_chall_readability_score(text)
     
def automated_readability(text):
    return textstat.automated_readability_index(text)

def flesch_ease(text):
    return textstat.flesch_reading_ease(text)

def flesch_kincaid(text):
    return textstat.flesch_kincaid_grade(text)

def gunning_fog(test_data):
    return textstat.gunning_fog(test_data)

def smog_index(test_data):
    return textstat.smog_index(test_data)

def overall(test_data):
    return textstat.text_standard(test_data)

    


