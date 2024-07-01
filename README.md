# CSNLP 2024: Course Project Group 15

Abstract: Large language models (LLMs) have substantially altered the landscape of machine learning in recent years. Besides mass adoption of chatbot-like interfaces powered by LLMs such as ChatGPT, a significant change is happening when it comes to modeling approaches for many natural language processing (NLP) tasks. First, the pre-trained LLMs have managed to provide text representations that appear to capture the language context better than ever before, thereby providing a much better foundation for fine-tuning on specific NLP tasks. Second, instruction-tuned LLMs have provided a completely new approach to managing NLP tasks via prompting, i.e., asking a generative model to perform an NLP task by simply generating a response. Both these LLM-based approaches have substantially reduced the labelled data requirement for achieving state-of-the-art performance, thereby dramatically reducing development time for accurate NLP models, which has massive practical implications. Hence, the question which naturally emerges for many NLP tasks is "which of the two approaches is better?". We answer this question for the case of information extraction from expense documents by comparing fine-tuned LLMs that are (i) trained via traditional NLP modeling approach for information extraction, (ii) trained to provide JSON-like output of extracted information based on a prompt. We evaluate the performance of several methods based on these two approaches on a case study using ~2000 labelled Swiss receipts, using data we obtained in cooperation with AI-startup nextesy AG. Our results show superior performance of the methods based on prompting instruction-tuned LLMs, as well as several other practically relevant implications that can guide decision making when developing information extraction systems for expense documents.

# Folders and scripts

## Folder data_processing:

- data_initial_extraction.csv is the file provided by partner company nextesy AG containing file names and corresponding extraction entities of interest: supplier name, supplier VAT number, date, and amount. 
- aws_textract.py contains the code for calling AWS Textract OCR service to extract the text from the raw documents.
- aws_textract.zip contains the .csv file (compressed due to size) which contains file names and text extractions from raw documents.
- data_cleaning.py contains the code for cleaning the examples such that they align with the task at hand, e.g., all extraction entities have to be contained in the extracted text.
- train_test_split.py contains the code for splitting the data in data_train.csv (~80%), data_test1.csv (~10%), and data_test2.csv (~10%), which are the main data files that were used in the analyses and can be found in folder data.

## Folder data:

- data_train.csv is the main training data file containing 1231 instances of receipts with corresponding columns: file name, extracted text, supplier name, supplier VAT number, date, and amount.
- data_test1.csv is the first testing set of the same format as data_train.csv, which contains 137 test instances of receipts from the same vendors that are found in data_train.csv (different documents but same document structure as training data).
- data_test2.csv is the second testing set of the same format as data_train.csv, which contains 144 test instances of receipts from different vendors that are not found in data_train.csv (different documents and different document structure compared to training data).
- instruct_data_k-shot.jsonl is a file containing train instances used for fine-tuning Mistral-7B with k-shot prompts, in the format required for the respective model fine-tuning.

## Folder prompts:

- prompt_k-shot.txt is a text file containing the prompt used for obtaining the results with LLM, with in-context learning using k examples.

## Folder code:

- data_prep_finetune.py is a script for preparing the data for fine-tuning Mistral-7B.
- mistral_finetune_models.py is a script to perform fine-tuning of Mistral models and deploy on Mistral platform.
- mistral_finetune_api.py is a script for obtaining the results of fine-tuned Mistral-7B model.
- aws_analyze_expense.py is a script for obtaining the results of AWS Analyze Expense API model.
- gpt_models.py is a script for obtaining the results of OpenAI LLMs GPT-3.5 Turbo and GPT-4 Turbo.
- llama_bedrock_api.py is a script for obtaining the results of Meta LLMs Llama3-8B and Llama3-70B.
- mistral_bedrock_api.py is a script for obtaining the results of Mistral LLMs Mistral-7B and Mixtral-8x7B.
- ExpBERT.ipynb is a script for implementation and obtaining the results of a BERT model for named entity recognition. 

## Folder results:

- results_NER.ipynb is a script for printing the results of NER-based approaches: AWS Analyze Expense API and BERT.
- results_LLMs.ipynb is a script for printing the results of prompt-based approaches: LLMs.
- the remaining files with .pickle extension are stored results from running scripts in folder code. 

