from version1 import extract_text, extract_exposure, csv_to_list, sentiment_score

def model5(analyze_path, exposure_csv, n):
    """
        Model 5 Pipeline:
        - text extraction from earnings call
        - exposure csv to exposure word list
        - exposure search with +- parameter
        - sentiment analysis on found exposure

        returns:
            dict with exposure strings, sentiment score, pos/neg

    """

    text = extract_text(analyze_path)
    exposure_word_list = csv_to_list(exposure_csv)
    exposure = extract_exposure(text, exposure_word_list, window = n)
    final = sentiment_score(exposure)

    return final

