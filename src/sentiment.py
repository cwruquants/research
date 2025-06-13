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
    Imports library with LM and HIV4 methodgis, tokenizes the string,
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
    import pysentiment2 as ps
    hiv4 = ps.HIV4()
    tokens = hiv4.tokenize(text)
    scores = hiv4.get_score(tokens)
    
    return scores['Polarity']

def HIV4_Subjectivity(text) -> int:
    """
    Imports library with LM and HIV4 methods, tokenizes the string,
    and returns the subjectivity score as per the HIV4 Dictionary,
    which is calculated as the number of positive words plus the number of negative words,
    divided by the sum of positive words, negative words, and neutral words. (score closer to
    1 is more subjective than a score close to 0)

    Parameters:
    - text (str): Input string to analyze.

    Returns:
    - int: Subjectivity score.
    """       
    import pysentiment2 as ps
    hiv4 = ps.HIV4()
    tokens = hiv4.tokenize(text)
    scores = hiv4.get_score(tokens)
    
    return scores['Subjectivity']  

def ML_Bigrams_Positive(string) -> int:
    """
    This function takes a string as input and returns the number of positive bigrams in the string.
    A positive bigram is defined as a word that present in the list of positive words.
    """
    import pandas as pd
    import re
    import unicodedata

    # Normalize and clean the input string
    string = unicodedata.normalize("NFKD", string)
    string = re.sub(r'[-/]', ' ', string)                # Convert hyphens/slashes to spaces
    string = re.sub(r'[^\w\s]', '', string)              # Remove punctuation
    words = string.lower().split()

    # Load and normalize bigrams from Excel
    df = pd.read_excel('Garcia_MLWords.xlsx', sheet_name='ML_positive_bigram', header=None)
    negative_bigrams = set(
        unicodedata.normalize("NFKD", str(bigram)).strip().lower()
        for bigram in df[0]
    )

    # Build bigrams from input string
    input_bigrams = [' '.join([words[i], words[i + 1]]) for i in range(len(words) - 1)]

    # Count matches
    count = sum(1 for bigram in input_bigrams if bigram in negative_bigrams)
    return count

def ML_Unigrams_Positive(string) -> int:
    import pandas as pd
    """
    This function takes a string as input and returns the number of positive unigrams in the string.
    A positive unigram is defined as a word that present in the list of positive words.
    """
    # Load sheet
    df = pd.read_excel('Garcia_MLWords.xlsx', sheet_name='ML_positive_unigram', header=None)

    # Access the first column (column 0)
    positive_unigrams = df[0].tolist()
    words = string.split()
    
    # Initialize a counter for positive bigrams
    count = 0
    
    # Iterate through the words and count positive bigrams
    for i in range(len(words) - 1):
        if words[i] in positive_unigrams and words[i + 1] in positive_unigrams:
            count += 1
            
    return count


# NOT DONE YET
def ML_Bigrams_Negative(string) -> int:
    """
    This function takes a string as input and returns the number of negative bigrams in the string.
    A negative bigram is defined as a word that present in the list of negative words.
    """
    import pandas as pd
    import re
    import unicodedata

    # Normalize and clean the input string
    string = unicodedata.normalize("NFKD", string)
    string = re.sub(r'[-/]', ' ', string)                # Convert hyphens/slashes to spaces
    string = re.sub(r'[^\w\s]', '', string)              # Remove punctuation
    words = string.lower().split()

    # Load and normalize bigrams from Excel
    df = pd.read_excel('Garcia_MLWords.xlsx', sheet_name='ML_negative_bigram', header=None)
    negative_bigrams = set(
        unicodedata.normalize("NFKD", str(bigram)).strip().lower()
        for bigram in df[0]
    )

    # Build bigrams from input string
    input_bigrams = [' '.join([words[i], words[i + 1]]) for i in range(len(words) - 1)]

    # Count matches
    count = sum(1 for bigram in input_bigrams if bigram in negative_bigrams)
    return count


def ML_Unigrams_Negative(string) -> int:
    import pandas as pd
    """
    This function takes a string as input and returns the number of negative unigrams in the string.
    A negative unigram is defined as a word that present in the list of negative words.
    """
    # Load sheet
    df = pd.read_excel('Garcia_MLWords.xlsx', sheet_name='ML_negative_unigram', header=None)

    # Access the first column (column 0)
    negative_unigrams = df[0].tolist()
    words = string.split()
    
    # Initialize a counter for positive bigrams
    count = 0
    
    # Iterate through the words and count positive bigrams
    for i in range(len(words) - 1):
        if words[i] in negative_unigrams and words[i + 1] in negative_unigrams:
            count += 1
            
    return count
