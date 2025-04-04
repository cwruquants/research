# Sentiment features from Chin et. al.
def LM_Positive(text) -> int:
    """
    Imports library with LM and HIV4 methods, tokenizes the string,
    and returns the number of positive words as per the Loughran-McDonald Dictionary.

    Parameters:
    - text (str): Input string to analyze.

    Returns:
    - int: Positive sentiment score.
    """
    import pysentiment2 as ps
    lm = ps.LM()
    
    # Tokenize the text using LM's tokenizer
    tokens = lm.tokenize(text)

    # Get the sentiment scores
    score = lm.get_score(tokens)

    return score['Positive']

def LM_Negative(text) -> int:
    """
    Imports library with LM and HIV4 methods, tokenizes the string,
    and returns the number of negative words as per the Loughran-McDonald Dictionary.

    Parameters:
    - text (str): Input string to analyze.

    Returns:
    - int: Negative sentiment score.
    """
    import pysentiment2 as ps
    lm = ps.LM()
    
    # Tokenize the text using LM's tokenizer
    tokens = lm.tokenize(text)

    # Get the sentiment scores
    score = lm.get_score(tokens)

    return score['Negative']   

def LM_net_sentiment(text) -> int:
    """
    Imports library with LM and HIV4 methods, tokenizes the string,
    and returns the net sentiment score as per the Loughran-McDonald Dictionary,
    which is calculated as the number of positive words minus the number of negative words.

    Parameters:
    - text (str): Input string to analyze.

    Returns:
    - int: net sentiment score.
    """

    import pysentiment2 as ps
    lm = ps.LM()
    
    # Tokenize the text using LM's tokenizer
    tokens = lm.tokenize(text)

    # Get the sentiment scores
    score = lm.get_score(tokens)

    #net sentiment = positive score - negative score
    net_sentiment = score['Positive'] - score['Negative']

    return net_sentiment

def LM_Polarity(text) -> int:
    """
    Imports library with LM and HIV4 methods, tokenizes the string,
    and returns the polarity score as per the Loughran-McDonald Dictionary,
    which is calculated as the number of positive words minus the number of negative words,
    divided by the difference of positive words minus negative words.

    Parameters:
    - text (str): Input string to analyze.

    Returns:
    - int: Polarity score.
    """

    import pysentiment2 as ps
    lm = ps.LM()
    
    # Tokenize the text using LM's tokenizer
    tokens = lm.tokenize(text)

    # Get the sentiment scores
    score = lm.get_score(tokens)

    return score['Polarity'] 

def LM_Subjectivity(text) -> int:
    """
    Imports library with LM and HIV4 methods, tokenizes the string,
    and returns the subjectivity score as per the Loughran-McDonald Dictionary,
    which is calculated as the number of positive words plus the number of negative words,
    divided by the sum of positive words, negative words, and neutral words. (score closer to
    1 is more subjective than a score close to 0)

    Parameters:
    - text (str): Input string to analyze.

    Returns:
    - int: Subjectivity sentiment score.
    """

    import pysentiment2 as ps
    lm = ps.LM()
    
    # Tokenize the text using LM's tokenizer
    tokens = lm.tokenize(text)

    # Get the sentiment scores
    score = lm.get_score(tokens)
  
    return score['Subjectivity']  

def HIV4_Positive(text) -> int:
    """
    Imports library with LM and HIV4 methods, tokenizes the string,
    and returns the number of positive words as per the Harvard IV4 Dictionary.

    Parameters:
    - text (str): Input string to analyze.

    Returns:
    - int: Positive sentiment score.
    """   
    import pysentiment2 as ps
    hiv4 = ps.HIV4()
    tokens = hiv4.tokenize(text)
    scores = hiv4.get_score(tokens)
    
    return scores['Positive']

def HIV4_Negative(text) -> int:
    """
    Imports library with LM and HIV4 methods, tokenizes the string,
    and returns the number of negative words as per the Harvard IV4 Dictionary.

    Parameters:
    - text (str): Input string to analyze.

    Returns:
    - int: Negative sentiment score.
    """   
    import pysentiment2 as ps
    hiv4 = ps.HIV4()
    tokens = hiv4.tokenize(text)
    scores = hiv4.get_score(tokens)
    
    return scores['Positive']

def HIV4_net_sentiment(text) -> int:
    """
    Imports library with LM and HIV4 methods, tokenizes the string,
    and returns the net sentiment score as per the Harvard IV4,
    which is calculated as the number of positive words minus the number of negative words.

    Parameters:
    - text (str): Input string to analyze.

    Returns:
    - int: Net sentiment score.
    """   
    import pysentiment2 as ps
    hiv4 = ps.HIV4()
    tokens = hiv4.tokenize(text)
    scores = hiv4.get_score(tokens)

    #net sentiment = positive score - negative score
    net_sentiment = scores['Positive'] - scores['Negative']
    
    return net_sentiment

def HIV4_Polarity(text) -> int:
    """
    Imports library with LM and HIV4 methods, tokenizes the string,
    and returns the polarity score as per the Harvard IV4 Dictionary,
    which is calculated as the number of positive words minus the number of negative words,
    divided by the difference of positive words minus negative words.

    Parameters:
    - text (str): Input string to analyze.

    Returns:
    - int: Polarity score.
    """   
    hiv4 = ps.HIV4()
    tokens = hiv4.tokenize(text)
    scores = hiv4.get_score(tokens)
    
    return scores['Polarity']

def HIV4_Subjectivity(text) -> int:
    """
    Imports library with LM and HIV4 methods, tokenizes the string,
    and returns the subjectivity score as per the Loughran-McDonald Dictionary,
    which is calculated as the number of positive words plus the number of negative words,
    divided by the sum of positive words, negative words, and neutral words. (score closer to
    1 is more subjective than a score close to 0)

    Parameters:
    - text (str): Input string to analyze.

    Returns:
    - int: Subjectivity score.
    """       
    hiv4 = ps.HIV4()
    tokens = hiv4.tokenize(text)
    scores = hiv4.get_score(tokens)
    
    return scores['Subjectivity']  