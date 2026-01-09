import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import random # Used for simulation only
import requests # Used for real ML API calls

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

# ==========================================
# 1. CONFIGURATION (EDIT THIS SECTION)
# ==========================================
# ⚠️ You must generate an App Password from your Google Account settings.
# Do not use your normal login password.
EMAIL_SENDER = "saikarthikreddykuppireddy@gmail.com"  # Your Email
EMAIL_PASSWORD = "mmci cuhh kwbn sdgo"   # Your App Password

# ==========================================
# 2. INITIALIZE APP & SCHEDULER
# ==========================================
app = FastAPI(title="Alerts Middleware API", version="2.0")

# Initialize the scheduler to run jobs in the background
scheduler = BackgroundScheduler()
scheduler.start()

# ==========================================
# 3. ML API INTEGRATION (The Input Source)
# ==========================================
def fetch_ml_signal(ticker: str):
    """
    Connects to the ML Team's API to get the latest signal.
    """
    print(f"🔍 [System] Connecting to ML API for {ticker}...")

    # Simulated Response (Replace with real API call)
    simulated_signal = random.choice(["BUY", "SELL", "HOLD"])
    simulated_price = round(random.uniform(100, 500), 2)

    return {
        "ticker": ticker,
        "signal": simulated_signal,
        "price": simulated_price,
        "confidence": "85%"
    }

# ==========================================
# 4. EMAIL ALERT LOGIC (The Output)
# ==========================================
def send_email_alert(user_email: str, ticker: str, signal_data: dict):
    """
    Sends a real email to the user using Gmail SMTP.
    """
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # HTML styled email body
        subject = f"🚨 ALERT: {signal_data['signal']} {ticker}"
        body = f"""
        Subject: {subject}

        TRADING ALERT SYSTEM
        ---------------------------------
        Time:     {current_time}
        Ticker:   {ticker}
        Signal:   {signal_data['signal']}
        Price:    {signal_data['price']}
        Confidence: {signal_data.get('confidence', 'N/A')}
        ---------------------------------
        Please check your dashboard for details.
        """

        # Setup the email
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_SENDER
        msg['To'] = user_email

        # Send via Gmail SSL
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, user_email, msg.as_string())

        print(f"✅ [Success] Email sent to {user_email} for {ticker}")

    except Exception as e:
        print(f"❌ [Email Failed] Could not send email: {e}")

# ==========================================
# 5. THE CORE JOB (Runs at scheduled times)
# ==========================================
def check_and_alert_job(user_email: str, ticker: str):
    """
    This function runs automatically at 10:00 AM, 12:30 PM, etc.
    """
    print(f"\n⏰ [Scheduler] Running check for {ticker} (User: {user_email})")

    # 1. Get Data from ML
    data = fetch_ml_signal(ticker)

    if not data:
        print("⚠️ No data received from ML API.")
        return

    # 2. Check Signal
    if data['signal'] in ["BUY", "SELL","HOLD"]:
        print(f"🚀 Signal Detected: {data['signal']}. Sending Email...")
        send_email_alert(user_email, ticker, data)
    else:
        print(f"No email sent. Error occured")

# ==========================================
# 6. API ENDPOINTS (For Dashboard Team)
# ==========================================

# Input Model: This is what the Dashboard sends you
class AlertRequest(BaseModel):
    user_email: str
    ticker_name: str

@app.post("/create-alert")
async def create_alert(request: AlertRequest):
    """
    Dashboard calls this to set up alerts for a user.
    Schedules 3 checks daily: 10:00 AM, 12:30 PM, 3:00 PM.
    """
    email = request.user_email
    ticker = request.ticker_name
    base_id = f"{email}_{ticker}"

    print(f"📥 [API Received] Request to alert {email} on {ticker}")

    try:
        # Schedule Job 1: 10:00 AM
        scheduler.add_job(check_and_alert_job, 'cron', hour=10, minute=0,
                          id=f"{base_id}_10am", args=[email, ticker], replace_existing=True)

        # Schedule Job 2: 12:30 PM
        scheduler.add_job(check_and_alert_job, 'cron', hour=12, minute=30,
                          id=f"{base_id}_1230pm", args=[email, ticker], replace_existing=True)

        # Schedule Job 3: 03:00 PM
        scheduler.add_job(check_and_alert_job, 'cron', hour=15, minute=00,
                          id=f"{base_id}_3pm", args=[email, ticker], replace_existing=True)

        return {
            "status": "success",
            "message": f"Alerts set for {ticker} at 10:00, 12:30, 15:00",
            "user": email
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/stop-alert/{user_email}/{ticker_name}")
async def stop_alert(user_email: str, ticker_name: str):
    """
    Stops the alerts for a specific user and ticker.
    """
    base_id = f"{user_email}_{ticker_name}"
    try:
        scheduler.remove_job(f"{base_id}_10am")
        scheduler.remove_job(f"{base_id}_1230pm")
        scheduler.remove_job(f"{base_id}_3pm")
        return {"status": "success", "message": f"Stopped alerts for {ticker_name}"}
    except Exception:
        return {"status": "warning", "message": "Job not found or already stopped"}

@app.get("/")
def health_check():
    return {"status": "active", "system": "Alerts Middleware API"}
