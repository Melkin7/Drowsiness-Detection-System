# send_alert.py

from twilio.rest import Client

# Twilio credentials (Use environment variables in production)
ACCOUNT_SID = 'AC5eef2f5ccf4b61888af3d8733cdb49e1'
AUTH_TOKEN = '425be39d6ee5fda1783b17602ff11bea'
TWILIO_PHONE_NUMBER = '+17744867982'          # Your Twilio number
EMERGENCY_CONTACT_NUMBER = '+919344415691'    # Emergency contact number

def send_sms_alert():
    """Send an SMS alert via Twilio"""
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        message = client.messages.create(
            body="🚨 Drowsiness detected! Please check the driver's condition.",
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_CONTACT_NUMBER
        )
        print(f"SMS sent. SID: {message.sid}")
    except Exception as e:
        print("Failed to send SMS:", e)

def make_call_alert():
    """Make a voice call alert via Twilio"""
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        call = client.calls.create(
            twiml='<Response><Say voice="alice">Alert! Drowsiness detected. Please check the driver immediately.</Say></Response>',
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_CONTACT_NUMBER
        )
        print(f"Call initiated. SID: {call.sid}")
    except Exception as e:
        print("Failed to make call:", e)
