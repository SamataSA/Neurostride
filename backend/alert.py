import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from twilio.rest import Client

# =====================
# EMAIL CONFIG
# =====================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"
RECEIVER_EMAIL = "security_team@example.com"

# =====================
# TWILIO CONFIG
# =====================
TWILIO_SID = "your_twilio_account_sid"
TWILIO_AUTH_TOKEN = "your_twilio_auth_token"
TWILIO_PHONE = "+1234567890"  # Twilio number
ALERT_PHONE = "+1987654321"   # Your phone

def send_email_alert(suspect_id, similarity, behaviors, image_path=None):
    try:
        subject = "🚨 ALERT: Suspicious Activity Detected"
        body = f"""
        Suspicious person detected.

        Suspect ID: {suspect_id if suspect_id else 'Unknown'}
        Similarity: {similarity:.2f}
        Suspicious Behaviors: {', '.join(behaviors) if behaviors else 'None'}
        """

        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
            msg.attach(part)

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print(f"[ALERT] Email sent to {RECEIVER_EMAIL}")
    except Exception as e:
        print(f"[ALERT ERROR] Email failed: {e}")

def send_sms_alert(suspect_id, similarity, behaviors):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        message = f"🚨 ALERT: Suspicious person detected!\nID: {suspect_id or 'Unknown'}\nSimilarity: {similarity:.2f}\nBehaviors: {', '.join(behaviors) if behaviors else 'None'}"
        client.messages.create(body=message, from_=TWILIO_PHONE, to=ALERT_PHONE)
        print(f"[ALERT] SMS sent to {ALERT_PHONE}")
    except Exception as e:
        print(f"[ALERT ERROR] SMS failed: {e}")

def send_alert(suspect_id, similarity, behaviors, image_path=None):
    send_email_alert(suspect_id, similarity, behaviors, image_path)
    send_sms_alert(suspect_id, similarity, behaviors)
