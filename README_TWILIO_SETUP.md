# SpectraGuard v2 - Twilio SMS Configuration Guide

Follow these steps to configure the official Twilio REST API notifications.

## Step 1: Set Account SID
Locate your Twilio Account SID on the Twilio Console Dashboard and add it to your `.env` file:
```env
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Step 2: Set Auth Token
Locate your Twilio Auth Token on the Twilio Console Dashboard and add it to your `.env` file:
```env
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Step 3: Set Twilio Number
Locate your Twilio virtual phone number and add it to your `.env` file (must be in E.164 format e.g. `+1234567890`):
```env
TWILIO_PHONE_NUMBER=+1234567890
```

## Step 4: Run Verification
Validate the REST API connection by running:
```powershell
python test_twilio.py
```
This script will verify your environment variables, initialize the Twilio client, and dispatch a test message.
