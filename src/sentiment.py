# Sentiment features from Chin et. al.
def LM_Positive(text) -> int:
    """

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
    Parameters:
    - text (str): Input string to analyze.

    Returns:
    - int: Subjectivity score.
    """       
    hiv4 = ps.HIV4()
    tokens = hiv4.tokenize(text)
    scores = hiv4.get_score(tokens)
    
    return scores['Subjectivity']  