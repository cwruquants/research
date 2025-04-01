# Readability features from Chin et. al.

# Readability features from Chin et. al.


import textstat

def coleman_liau(text):
    """Calculates the Coleman-Liau index.The Coleman-Liau Index is a readability test designed to estimate the U.S. grade level needed to understand a given text
    
    Args:
        text (str): The input text to evaluate.
    
    Returns:
        float: The Coleman-Liau index score.
    """
    return textstat.coleman_liau_index(text)

def dale_chall(text):
    """Computes the Dale-Chall readability score. The Dale-Chall Readability Index is a readability formula that assesses how difficult a text is to understand.
       Unlike some other readability tests, it considers the familiarity of words by comparing them against a list of 3,000 common words
    
    Args:
        text (str): The input text to evaluate.
    
    Returns:
        float: The Dale-Chall readability score.
    """
    return textstat.dale_chall_readability_score(text)
     
def automated_readability(text):
    """Calculates the Automated Readability Index (ARI). The Automated Readability Index (ARI) is a readability test that estimates the U.S. grade level needed to understand a text.
       It's similar to the Coleman-Liau Index but is based on characters per word and words per sentence.

    Args:
        text (str): The input text to evaluate.
    
    Returns:
        float: The ARI score.
    """
    return textstat.automated_readability_index(text)

def flesch_ease(text):
    """Computes the Flesch Reading Ease score. The Flesch Reading Ease Score (FRES) is a readability test that measures how easy a text is to read. 
       It is widely used in education, publishing, and government documents to ensure content is accessible to the target audience
    
    Args:
        text (str): The input text to evaluate.
    
    Returns:
        float: The Flesch Reading Ease score.
    """
    return textstat.flesch_reading_ease(text)

def flesch_kincaid(text):
    """Calculates the Flesch-Kincaid Grade Level. The Flesch-Kincaid Grade Level Index is a readability test that indicates the U.S. school grade required to understand a text. 
      It is based on words per sentence and syllables per word.

    Args:
        text (str): The input text to evaluate.
    
    Returns:
        float: The Flesch-Kincaid Grade Level.
    """
    return textstat.flesch_kincaid_grade(text)

def gunning_fog(test_data):
    """Computes the Gunning Fog Index. The Gunning Fog Index is a readability metric that estimates the number of years of formal education required to understand a given text. 
       It emphasizes sentence length and complex words (words with three or more syllables), making it useful for evaluating business, legal, and academic writing.
    
    Args:
        test_data (str): The input text to evaluate.
    
    Returns:
        float: The Gunning Fog Index score.
    """
    return textstat.gunning_fog(test_data)

def smog_index(test_data):
    """Calculates the SMOG Index. The SMOG Index (Simple Measure of Gobbledygook) is a readability formula that estimates the years of education required to understand a text.
       It focuses on polysyllabic words (words with three or more syllables) to determine readability.
    
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
    
    


