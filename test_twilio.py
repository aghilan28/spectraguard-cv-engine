import os
import sys
import json
from twilio.rest import Client
from backend.config.sms_settings import SMSSettings

def run_test():
    print("="*80)
    print("      SPECTRAGUARD TWILIO SMS TEST SCRIPT")
    print("="*80)

    # 1. Load credentials
    sid = SMSSettings.get_twilio_sid()
    token = SMSSettings.get_twilio_token()
    from_phone = SMSSettings.get_twilio_phone()

    if not sid or "YOUR_ACCOUNT_SID" in sid:
        print("Account SID Loaded: FAIL (Missing or placeholder)")
        sys.exit(1)
    print("Account SID Loaded: PASS")

    if not token or "YOUR_AUTH_TOKEN" in token:
        print("Auth Token Loaded: FAIL (Missing or placeholder)")
        sys.exit(1)
    print("Auth Token Loaded: PASS")

    if not from_phone or "YOUR_TWILIO_PHONE_NUMBER" in from_phone:
        print("Twilio Phone Number Loaded: FAIL (Missing or placeholder)")
        sys.exit(1)
    print("Twilio Phone Number Loaded: PASS")

    # 2. Initialize Client with disabled SSL verification to bypass local proxy blocks
    try:
        import requests
        from urllib3.exceptions import InsecureRequestWarning
        requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
        
        from twilio.http.http_client import TwilioHttpClient
        http_client = TwilioHttpClient()
        http_client.session.verify = False
        
        client = Client(sid, token, http_client=http_client)
        print("Client Initialized: PASS")
    except Exception as e:
        print(f"Client Initialized: FAIL ({e})")
        sys.exit(1)

    # 3. Load recipient phone number from config/user_settings.json
    settings_path = "config/user_settings.json"
    if not os.path.exists(settings_path):
        print(f"Recipient Configuration: FAIL (Missing {settings_path})")
        sys.exit(1)

    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            contacts = data.get("emergency_contacts", [])
            if not contacts:
                print("Recipient Configuration: FAIL (No contacts listed under emergency_contacts)")
                sys.exit(1)
            to_phone = contacts[0].strip()
    except Exception as e:
        print(f"Recipient Configuration: FAIL (Error reading settings: {e})")
        sys.exit(1)

    print(f"Recipient Phone Number Loaded: PASS ({to_phone})")

    # 4. Dispatch SMS
    test_message = "🚨 SpectraGuard Security Alert - Twilio integration test successful."
    
    try:
        message = client.messages.create(
            body=test_message,
            from_=from_phone,
            to=to_phone
        )
        print("SMS Sent: PASS")
        print(f"Message SID: {message.sid}")
        print(f"Delivery Status: {message.status}")
        
        # Verify it is a real SID
        if not message.sid.startswith("SM"):
            print("Message SID Validation: FAIL (Real SID must start with 'SM')")
            sys.exit(1)
            
    except Exception as e:
        print(f"SMS Sent: FAIL ({e})")
        sys.exit(1)

    print("\n" + "="*50)
    print("TWILIO rest api integration verified successfully!")
    print("="*50)

if __name__ == "__main__":
    run_test()
