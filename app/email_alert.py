import smtplib
from email.mime.text import MIMEText

def send_email_alert(category, views):
    sender_email = "testsender123@gmail.com"   # Dummy sender
    receiver_email = "devrarahul666@gmail.com"  # Your email
    password = "yourapppassword"  # App password needed (don't use your real password)

    subject = f"🚀 Trending Alert: {category} Category!"
    body = f"The forecasted views for {category} are {views}. It's trending highly! 🚀"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("✅ Email alert sent successfully!")
    except Exception as e:
        print(f"❌ Error sending email: {e}")
