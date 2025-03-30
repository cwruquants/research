# Readability features from Chin et. al.

# Readability features from Chin et. al.


import textstat

def coleman_liau(text):
    """Calculates the Coleman-Liau index
    
    Args:
        text (str): The input text to evaluate.
    
    Returns:
        float: The Coleman-Liau index score.
    """
    return textstat.coleman_liau_index(text)

def dale_chall(text):
    """Computes the Dale-Chall readability score
    
    Args:
        text (str): The input text to evaluate.
    
    Returns:
        float: The Dale-Chall readability score.
    """
    return textstat.dale_chall_readability_score(text)
     
def automated_readability(text):
    """Calculates the Automated Readability Index (ARI)

    Args:
        text (str): The input text to evaluate.
    
    Returns:
        float: The ARI score.
    """
    return textstat.automated_readability_index(text)

def flesch_ease(text):
    """Computes the Flesch Reading Ease score
    
    Args:
        text (str): The input text to evaluate.
    
    Returns:
        float: The Flesch Reading Ease score.
    """
    return textstat.flesch_reading_ease(text)

def flesch_kincaid(text):
    """Calculates the Flesch-Kincaid Grade Level

    Args:
        text (str): The input text to evaluate.
    
    Returns:
        float: The Flesch-Kincaid Grade Level.
    """
    return textstat.flesch_kincaid_grade(text)

def gunning_fog(test_data):
    """Computes the Gunning Fog Index
    
    Args:
        test_data (str): The input text to evaluate.
    
    Returns:
        float: The Gunning Fog Index score.
    """
    return textstat.gunning_fog(test_data)

def smog_index(test_data):
    """Calculates the SMOG Index
    
    Args:
        test_data (str): The input text to evaluate.
    
    Returns:
        float: The SMOG Index score.
    """
    return textstat.smog_index(test_data)

def overall(test_data):
    """Caculates overall readability
    
    Args:
        test_data (str): The input text to evaluate.
    
    Returns:
        str: The estimated grade level required to understand the text.
    """
    return textstat.text_standard(test_data)
    
    


