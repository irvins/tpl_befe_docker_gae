import os
import json
import re
import numpy as np
import nltk
import openai
import difflib
import requests
import time
import types
import pprint
import math
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from datetime import datetime, timezone

import logging
from . import logging_setup

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set your API key
openai.api_key = os.environ.get("OPENAI_API_KEY")
from google.cloud import firestore
from .firestore_utils import get_firestore_client
db = get_firestore_client()


#NEED TO BUILDTHE API CALL IN ORDER TO SET THE timeou
def create_completion_with_timeout(engine, prompt, max_tokens, n, stop, temperature, timeout):
    url = f'https://api.openai.com/v1/engines/{engine}/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai.api_key}'
    }

    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "n": n,
        "stop": stop,
        "temperature": temperature,
    }
    

    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)

    if response.status_code != 200:
        logging.debug(f"Error {response.status_code}: {response.text}")
        return None

    return response.json()


#basic chatgpt completion with davinci 3
def basic_completion(designed_prompt, max_response_tokens=3100):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=designed_prompt,
        max_tokens=max_response_tokens,
        n=1,
        stop=None,
        temperature=0.5,
    )

    if response.choices:
        return response.choices[0].text.strip()
    else:
        return "Sorry, I don't know"


#using NLTK , tokenize and normalize user input AND context raw data
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    stemmer    = PorterStemmer()
    tokens     = word_tokenize(text)
    tokens     = [token.lower() for token in tokens if token.isalnum()]
    tokens     = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return tokens


#get potential contexts from firestore, currently getting all but will need a way to filter as data set gets largerz
def fetch_context_data_from_firestore():
    # maybe try to do an initial query that doesnt require getting EVERYTHING

    docs = db.collection("contexts").get()

    # Extract document data as list of dictionaries
    context_data = [doc.to_dict() for doc in docs]
    
    return context_data


#use Scikit-learn to find best context match between user input and context data
def find_most_relevant_context(preprocessed_user_input_keywords):
    most_relevant_context   = {}
    similarity_scores       = []
    best_match_index        = None

    # Fetch context data from Firestore
    context_data = fetch_context_data_from_firestore()

    # Preprocess context data
    for i in range(len(context_data)):
        context_data[i]['preprocessed_sections'] = []
        for section in context_data[i]['sections']:
            preprocessed_question = " ".join(preprocess_text(section['question']))
            preprocessed_answer = " ".join(preprocess_text(section['answer']))
            context_data[i]['preprocessed_sections'].append({
                'preprocessed_question': preprocessed_question,
                'preprocessed_answer': preprocessed_answer
            })

    preprocessed_context_data = []
    map_back = {}
    n = 0
    for i in range(len(context_data)):
        for section in context_data[i]['sections']:
            preprocessed_question = " ".join(preprocess_text(section['question']))
            preprocessed_answer = " ".join(preprocess_text(section['answer']))
            preprocessed_context = f"{preprocessed_question}\n{preprocessed_answer}"
            map_back[n] = i
            preprocessed_context_data.append(preprocessed_context)
            n += 1

    
    # Vectorize context data and user input keywords
    vectorizer                = TfidfVectorizer()
    
    if len(context_data) > 0:
        context_data_tfidf        = vectorizer.fit_transform(preprocessed_context_data)
        user_input_keywords_tfidf = vectorizer.transform([" ".join(preprocessed_user_input_keywords)])

        # Calculate similarity scores
        similarity_scores         = cosine_similarity(user_input_keywords_tfidf, context_data_tfidf)

        # Set a similarity threshold
        similarity_threshold      = 0.1 #anything less than this will be ignored, but can be manually fine tuned later.
        max_score                 = np.max(similarity_scores)

        if max_score > similarity_threshold:
            # Find the most relevant context
            best_match_index = np.argmax(similarity_scores)
        else:
            best_match_index = -1

        if 0 <= best_match_index < len(preprocessed_context_data):
            mapped_index = map_back[best_match_index]
            most_relevant_context = context_data[mapped_index]
        else:
            # Handle the case when there is no valid index found
            most_relevant_context = "{}"


    extra_info = {"most_relevant_context" : most_relevant_context, "similarity_scores" : similarity_scores, "best_match_index" : best_match_index}

    return extra_info


#find exact match from cached responses
def found_cache_match(preprocessed_user_input_keywords, similarity_threshold=85):
    found_match = None 
    
    collection_name = "cached_responses"
    property_name_1 = "tokenized_input"
    property_name_2 = "rating"
    array_to_match = preprocessed_user_input_keywords
    second_property_value = 1
    
    # Create the query for the second property
    query = db.collection(collection_name).where(property_name_2, "==", second_property_value)
    
    # Execute the query using this "generator"
    # documents = query.stream()

    documents = list(query.stream())
    """
    Please note that converting the generator to a list will store all the DocumentSnapshot objects in memory. This might not be an issue for small collections, but for large collections, it can consume a significant amount of memory. If you need to process a large number of documents, it's usually more efficient to work with a generator and process the documents one at a time as you iterate through them.
    """

    # Filter the documents based on the full array match
    matching_documents = [doc for doc in documents if doc.to_dict()[property_name_1] == array_to_match]

    # Print the document IDs of the matching documents
    found_match = None 
    if len(documents) > 0:
        for doc in matching_documents:
            #TODO need to think about this for multiple matches which could happen? maybe not actually.. since all future matches will default to the same answer
            #BUT those tokens are matched with a context,  will the context always match the same... yes actually.  wtf
            found_match = doc
            break

        if not found_match:
            logging.debug("no match found lets do fuzzy matches then")
            
            # Calculate similarity scores for each array in search_list
            similarity_scores = {average_similarity(preprocessed_user_input_keywords, doc.to_dict()[property_name_1]): doc for doc in documents}
            
            # Get the highest similarity score and its corresponding array
            max_similarity_score = max(similarity_scores, key=float)
            best_match = similarity_scores[max_similarity_score]

            # since this is pulling from cache, might want to increase the threshold 85 seems to be ok to allow through "big cat" vs "cat" , but not "lion" vs "cat"
            if max_similarity_score > similarity_threshold :
                found_match = best_match
        
    return found_match


# Custom function to calculate the average similarity between two arrays of different lengths
def average_similarity(array1, array2):
    total_similarity = 0
    matches = 0

    for item1 in array1:
        best_similarity = 0
        for item2 in array2:
            similarity = fuzz.ratio(item1, item2)
            if similarity > best_similarity:
                best_similarity = similarity
        total_similarity += best_similarity
        matches += 1

    for item2 in array2:
        best_similarity = 0
        for item1 in array1:
            similarity = fuzz.ratio(item2, item1)
            if similarity > best_similarity:
                best_similarity = similarity
        total_similarity += best_similarity
        matches += 1

    logging.debug("average similarity score for following two lists")
    logging.debug(array1)
    logging.debug(array2)
    logging.debug(total_similarity / matches)
    return total_similarity / matches if matches > 0 else 0


# THERE IS A NEW "STYLE" Prompt DESIGN THAT SOUNDED DOPE BUT STILL NEEDS TO BE CONVERTED TO RAW TEXT SO IT SUCKS
def systemRole(title, summary):
    #if context present, then this should be the first item in the "messages" array  passed to the api as the prompt along with the user input
    content = f"You are an assistant in a medical institution with knowledge about {title} : {summary}"
    system_role = {"role": "system", "content" : content}

    return system_role


def userRole(user_input):
    #finally append the most recent user's raw input as the last item in the array
    user_role = {"role": "user", "content": user_input};

    return user_role


def assistantRole(content):
    #finally append the most recent user's raw input as the last item in the array
    assistant_role = {"role": "assistant", "content": content};

    return assistant_role


def simulatedAssistantRole(sections):
    #if context present and in Q and A format, then simulate an ongoing conversation in the "messages" array between "user" role and "asssitant" roll
    simulated_context = []
    for section in sections:
        simulated_context.append(userRole(section["question"]))
        simulated_context.append(assistantRole(section["answer"]))

    return simulated_context


def getFormattedMsgString(msg):
    return f"{msg['role']}: {msg['content']}\n"


def messageStylePromptDesign(user_input, previous_prompt=""):
    preprocessed_user_input_keywords    = preprocess_text(user_input)
    relevant_context_data               = find_most_relevant_context(preprocessed_user_input_keywords)

    # lets concat the relevant_context_data to the prompt
    context_dict    = relevant_context_data["most_relevant_context"]
    context_id      = ""
    messages        = []

    if context_dict and not previous_prompt: 
        #if ther is context, then use "title" and "description" property to make the single "system" role
        #have at least one to tell HOW chatGPT should answer (ie, in the style of Dr.Suess, or return output in this json structure example {data: {}}
        system_role = systemRole(context_dict["title"], context_dict["summary"])
        messages.append(system_role)

        #if there is context it will be in Q&A format in "sections", create simulated conversation between "user" role and "assistant" role
        context_id = context_dict["id"]
        assistant_role = simulatedAssistantRole(context_dict['sections'])
        messages.extend(assistant_role)

        previous_prompt = ""
        
    #the last item in the array should be the current user_input, that the API should specifically answer
    messages.append(userRole(user_input))

    #fucking me up here chatgpt, even with messaeg format still needs to feed raw text to api prompt
    formatted_string = "".join([getFormattedMsgString(msg) for msg in messages])
  
    new_prompt = previous_prompt + formatted_string

    #if this is not the first query in the session, then keep growing the previous prompt (possibly adding a new "system" role for context)
    #use the user votes (possibly block query UI if they do not VOTE), to exclude "user/assistant" q+a if downvoted 
    return {"messages" : new_prompt, "context_id" : context_id}


#THE OLD STYLE RAWTEXT IS  AT LEAST MORE CONCISE AND SKIPS THE MIDDLES STEP OF CREATEIONG AN ARRAY FIRST actu
def rawQA(q, a):
    return {"question" : q , "answer" : a}


def preparePreviousRawContext(rawText):
    raw_text_prompt = rawText
    lines = raw_text_prompt.split("\n")
    lines = [line for line in lines if not line.startswith("AI:")]
    cleaned_prompt = "\n".join(lines)
    
    #uh what am i doing here?
    # lines = raw_text_prompt.split("\n")
    # cleaned_lines = []

    # for line in lines:
    #     if line.strip() == "":
    #         break
    #     if not line.startswith("AI:"):
    #         cleaned_lines.append(line)

    # cleaned_prompt = "\n".join(cleaned_lines)
    return cleaned_prompt + "\n"


def getFormattedRawString(msg, response_only = False):
    if response_only:
        return f"{msg['answer']}\n\n"
    else: 
        return f"{msg['question']}:\n{msg['answer']}\n\n"


def getFormattedRawContextSTring(content, context_id):
        return f"Context {context_id}:\n{content}\n\n"


def rawPromptDesign(key, user_input, previous_prompt=None):
    rawPrompt = ""

    if not previous_prompt:
        previous_prompt = ""

    # old way of providing context with raw text context
    # returns context python dict, so do we not need to json.dumps?
    # preprocessed_user_input_keywords    = preprocess_text(user_input)
    # extra_info                          = find_most_relevant_context(preprocessed_user_input_keywords)    
    # relevant_context_data               = json.dumps(extra_info["most_relevant_context"])

    # lets concat th relevant_context_data to the prompt
    # context_dict        = json.loads(relevant_context_data)
    formatted_context   = ""
    context_id          = ""

    # if 'sections' in context_dict:
    #     context_id      = context_dict["id"]    

    #     logging.debug("Found a context, check if its already in the previous prompt befoer appending it to the whole thing")
    #     logging.debug(f"Context {context_id}:")

    #     #lets try using the context in paragraph form   
    #     #only include if not already in the previous_prompt
    #     #this seems way more concise and economical with the token usage, and gives more control about layering on more context with subsequent questions
    #     if f"Context {context_id}:" not in previous_prompt:
    #         logging.debug("only if thre IS a prev prompt and context id not in it, OR prev prompt is none")
    #         logging.debug(context_id)
    #         formatted_context += getFormattedRawContextSTring(context_dict["summary"], context_id)

    #     #lets use the context in chatgpt generated Q&A format
    #     # for section in context_dict['sections']:
    #     #     formatted_context += getFormattedRawString(section)
            
    # make sure to add a context line like Context : you are an sms virtual assistant, answering questions for clients pertaining to their real estate contracts.  please keep your responses less than 1600 characters

    #add the latest user input to the prompt package
    rawPrompt = f"{formatted_context}\n{user_input}\nAI:"
    new_prompt = previous_prompt + rawPrompt

    #get list of context ids used in this prompt for analysis later
    # pattern         = r'Context (.*):'
    # context_id_list = re.findall(pattern, new_prompt)
    context_id_list = []

    return {"prompt_w_context" : new_prompt, "context_id" : context_id_list}


#fetch a URL parse it with Beauifulsoup pull raw text content from specific #div_id
def scrape_content(url, div_id):
    response    = requests.get(url)

    soup        = BeautifulSoup(response.text, 'html.parser')

    content     = soup.find('div', {'id': div_id})

    if content:
        return content.get_text(strip=True)
    else:
        return None


#format raw text extracted from martin's .pages -> .docx 
#wont be the same goingforward if i give him custom input method  
def format_raw_text_to_context(raw_text):
    # Define the prompt to extract the relevant information
    prompt = f"""
    I have a list of important dates and events, along with some notes. I would like this information to be organized into a structured JSON format. The JSON should have two sections: 'Schedule' and 'Notes'. The 'Schedule' section should be a list of events, where each event has a 'date' and 'event'. The 'Notes' section should be a list of individual notes. Here is the text I need structured:
    
    {raw_text}
    """

    #set max tokens to 3200, reserve about 900 for the prompt
    # davinci 3 is slow as shit,  for this purpose use davinci-002
    response = create_completion_with_timeout(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=3100,
        n=1,
        stop=None,
        temperature=0.5,
        timeout=60
    )

    # string representation of json object , but not actually json yet
    formatted_text          = response["choices"][0]["text"].strip() if response and "choices" in response else ""

    # Remove the escape characters from the string
    formatted_text_clean    = formatted_text.replace("\\n", "").replace("\\", "")

    json_formatted_text     = json_formatted_text = json.loads(formatted_text_clean) if formatted_text_clean else {}

    logging.info("json_formatted_text text")
    logging.info(json_formatted_text)

    # store the new context in Firestore
    if formatted_text_clean:
        doc_ref = db.collection('escrow_schedules').document("4158469192")
        doc_ref = doc_ref.set(json_formatted_text) 

    return json_formatted_text


def clean_phone_number(country_code_phone_number):
    # Remove the '+' and country code
    return country_code_phone_number.replace("+1", "")


def is_within_last_hour(timestamp):
    return (time.time() - timestamp.timestamp()) < 3600  # Check if timestamp is within last hour


def handle_document(doc_genr8r, document_type):
    if isinstance(doc_genr8r, firestore.DocumentSnapshot):
        return doc_genr8r

    elif isinstance(doc_genr8r, types.GeneratorType):
        for doc in doc_genr8r:
            if doc.exists:  # Check if the document exists
                logging.error(f"{document_type} document exists!")
                return doc
    return None
     

def create_context_from_json(context_json):
    context = "Context:\n"
    context += "634 Pilgrim Dr, Foster City SCHEDULE:\n"
    
    for item in context_json["Schedule"]:
        context += item["date"] + " " + item["event"] + "\n"
    
    context += "NOTES:\n"
    
    for note in context_json["Notes"]:
        context += note + "\n"
    
    context += "\n"

    return context


def process_incoming_message(post_dict):
    phone_number    = clean_phone_number(post_dict.get('From'))
    message_body    = post_dict.get('Body')
    message_sid     = post_dict.get('MessageSid')
    
    transaction     = db.transaction()

    # define transaction function
    @firestore.transactional
    def transaction_fn(transaction):
        # Get the sms_conversations and escrow_schedules documents (if exist)
        conversation_ref    = db.collection('sms_conversations').document(phone_number)
        conversation_genr8r = transaction.get(conversation_ref)
        schedule_ref        = db.collection('escrow_schedules').document(phone_number)
        schedule_genr8r     = transaction.get(schedule_ref)
        
        current_time        = datetime.now(timezone.utc)
        
        conversation_doc    = handle_document(conversation_genr8r, "conversation_doc")
        schedule_doc        = handle_document(schedule_genr8r, "schedule_doc")
        contract_context    = None

        assistant_name      = "Aiko"

        formatted_previous  = f"Context: You are an SMS virtual assistant named {assistant_name}, answering questions from clients about their real estate contracts. The topics could include contract terms, negotiation processes, property details, and more. Please keep your responses concise and under 1600 characters.\n\n"

        # Check if a schedule document exists
        if schedule_doc is not None and schedule_doc.exists:
            # Convert the schedule document to a dictionary
            schedule_dict = schedule_doc.to_dict()
            
            # ADD CONTRACT CONTEXT TO THE PROMPT
            contract_context    = create_context_from_json(schedule_dict)
            formatted_previous += contract_context

        # Check if this is a new conversation or a continuation
        if conversation_doc is None or not conversation_doc.exists or not is_within_last_hour(conversation_doc.to_dict().get('last_message_time', None)):
            # Create a new conversation document
            conversation_data = {
                'last_message_time': current_time,
                'messages': []
            }
            transaction.set(conversation_ref, conversation_data)
        else:
            # Append the new message to the conversation
            conversation_data = conversation_doc.to_dict()
            #get the most recent 2 sets of incoming and outgoing  to use as previous context?  
            # add the previous incoming outgoing messages as part of context  

            # Get the last items, 2 minimum, 4 maximum
            last_four_items = conversation_data["messages"][-4:] if len(conversation_data["messages"]) >= 4 else conversation_data["messages"][-2:]

            formatted_previous += "Last couple Q&A Context:\n"
            for i in range(0, len(last_four_items), 2):
                formatted_previous += f"Q: {last_four_items[i]['text']}\n"
                formatted_previous += f"A: {last_four_items[i+1]['text']}\n"
                  
        # SET THE PROMPT AND CALL API
        ai_prompt   = rawPromptDesign(phone_number, message_body, formatted_previous)
        raw_prompt  = ai_prompt["prompt_w_context"]; 

        guesstimate_tokens_used = math.ceil((len(raw_prompt)/4) / 100) * 100
        response_message        = basic_completion(raw_prompt, 4000 - guesstimate_tokens_used)

        # Add the new incoming message AND outgoing message to the conversation document
        conversation_data['messages'].append({
            'direction'     : 'incoming',
            'message_sid'   : message_sid,
            'text'          : message_body,
            'timestamp'     : current_time
        })
        conversation_data['messages'].append({
            'direction': 'outgoing',
            'message_sid': message_sid,
            'text': response_message,
            'timestamp': datetime.now(timezone.utc)
        })
            
        # Update the last message time and messages
        update_data = {
            'last_message_time': current_time,
            'messages': firestore.ArrayUnion(conversation_data['messages'])
        }
        transaction.update(conversation_ref, update_data)

        return {"conversation_data" : conversation_data, "raw_prompt" : raw_prompt, "contract_context" : contract_context, "reply_msg" : response_message, "approximate_prompt_tokens" : guesstimate_tokens_used}
        
    # Run the transaction
    conversation_data = transaction_fn(transaction)

    return conversation_data
