from .. import app
import os
import logging
from flask import jsonify
from datetime import datetime

import openai
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from app.services.firestore_utils import get_firestore_client


@app.route('/voice')
def voice():
    logging.info('hey shithead this is the default route for "voice"')
    return f'try /openai_test, /firestore_test and /twilio_test'


@app.route('/openai_test', methods=['GET'])
def openai_test():
    logging.critical('hey bitchface this is openai_test')

    #test openai connection

    # Define the prompt to extract the relevant information
    prompt = f"""
    Can you tell me a random fact?
    AI:
    """
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=2048,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # string representation of json object , but not actually json yet
    formatted_text = response["choices"][0]["text"].strip() if "choices" in response else ""

    return f'openai test {formatted_text}'


@app.route('/firestore_test', methods=['GET'])
def firestore_test():
    logging.critical('hey shithead this is firestore_test')

    # Get the current timestamp
    timestamp = datetime.now()
    formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    # test firestore connection
    db = get_firestore_client()
    doc_ref = db.collection('template_test').document("4158469192")
    doc_ref = doc_ref.set({"message": f"{formatted_timestamp}"}) 
    return f'firestore test {formatted_timestamp}'


@app.route('/twilio_test', methods=['GET'])
def twilio_test():
    logging.critical('hey fuck face this is twilio_test')

    #test twilio connection

    # Your Twilio account SID and auth token, which you can find on your Twilio dashboard
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token  = os.environ.get("TWILIO_AUTH_TOKEN")

    # Initialize the Twilio client
    client      = Client(account_sid, auth_token)

    # Get the current timestamp
    timestamp = datetime.now()
    formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")


    # Your Twilio phone number, the recipient phone number, and the message you want to send
    from_phone_number   = "+13239919192"  # Your Twilio phone number (formatted with the country code)
    to_phone_number     = "+14158469192"    # The recipient's phone number (formatted with the country code)
    message_text        = f"Hello, this is a test message from Sophia! {formatted_timestamp}"

    try:
        message = client.messages.create(
            body=message_text,
            from_=from_phone_number,
            to=to_phone_number
        )
        return jsonify({'success': True, 'message_sid': message.sid})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
