--April Something, 2025--
Ethan/Professor Bae
* get rid fo KeyBERT, replace with cosine similarity function

--April 10 2025--
* Ethan finished making vault script... finally.

--April 1 2025--
* Karel finished HIV4 functions
* Anthony finished Risk function

--March 29 2025--
* Karel finished LM sentiment functions
* Charan finished readability functions
* Assigned Karel, Charan, Binayek to write paper
* Assigned Anthony to work on risk KeyBERT function, requirements document

Notes for meeting with Prof. Bae:
- using political bigrams from the paper did not have any significant difference
- added risk word function, currently working on debugging issues with KeyBERT dependencies

Ethan | Prof Bae. 
# TODO:
1. 3 way decomposition: Presentation, Q&A Executive, Q&A Analyst
- expand the existing readability and sentiment features on these "subsections"
2. Create a "Transcript Sample" folder in the Google Drive
- include the first 100 transcript SAMPLES in the folder
- in the code folder:
    - put code for the first 100 transcripts^
    - output of the code should be CSV with ID + all the features that we have calculated
    - include both CSV files for political
3. SET SCHEDULE FOR THE END OF SEMESTER PROJECT SHOWCASE
- 

FUTURE:
- replace political words and political bigrams with some sort of political BERT model


--March 19 2025--
Ethan | Prof Bae. 
# TODO:
1. risk count, risk percentage
3. run the bigrams specifically from the paper (/research/citations/input/political_bigrams/political_bigrams.csv) (comparison between bigram and non-bigram trial)

* current goal: quarterly rebalancing portfolio 
* stressed importance of keeping track of progress for the future

--March 5 2025--
Ethan: Ran KeyBERT function with same parameters as trial1, received some different results. Finished company extraction function. 

--February 28 2025 GB Meeting--
Karel: Finished KeyBERT function
Charan: Finished tfidf function
Diliya: Assigned to finish risk discovery function
Binayek: Assigned to extraction for company name, earnings call date, information
Atharva: Assigned to new task of web scraping

--February 14 2025 GB Meeting--
Progress:
- Karel: Short report on KeyBERT for more efficient text extraction
- Charan: Finished basic tf*idf function
- Diliya: Provided short summary of the RoBERTa model
- Ethan: Finished script to run model5 on folder of transcripts