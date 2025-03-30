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

    # Extract the positive score
    positive_score = score['Positive']

    return positive_score

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

    # Extract the negative score
    negative_score = score['Negative']  
    return negative_score  

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
    - int: Negative sentiment score.
    """

    import pysentiment2 as ps
    lm = ps.LM()
    
    # Tokenize the text using LM's tokenizer
    tokens = lm.tokenize(text)

    # Get the sentiment scores
    score = lm.get_score(tokens)

    # Extract the polarity score
    polarity_score = score['Polarity']  
    return polarity_score

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

    # Extract the polarity score
    subjectivity_score = score['Subjectivity']  
    return subjectivity_score
