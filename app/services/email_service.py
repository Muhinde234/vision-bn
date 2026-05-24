"""
Email delivery service.

Sends transactional emails via SMTP.
In development (SMTP_HOST not set), logs the email content instead of sending.
"""
import asyncio
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.config import settings
from app.core.logging import logger


def _send_sync(to_email: str, subject: str, html_body: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_FROM_EMAIL}>"
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        if settings.SMTP_USER:
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        server.sendmail(settings.SMTP_FROM_EMAIL, to_email, msg.as_string())

    logger.info("Email sent", to=to_email, subject=subject)


async def _send(to_email: str, subject: str, html_body: str) -> None:
    if not settings.SMTP_HOST:
        logger.info(
            "SMTP not configured — email skipped (set SMTP_HOST to enable)",
            to=to_email,
            subject=subject,
        )
        return
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _send_sync, to_email, subject, html_body)
    except Exception as exc:
        logger.error("Email delivery failed", to=to_email, subject=subject, error=str(exc))


async def send_password_reset_email(to_email: str, reset_link: str) -> None:
    html = f"""
    <div style="font-family:sans-serif;max-width:560px;margin:auto">
      <h2>VisionDx – Password Reset</h2>
      <p>You requested a password reset. Click the button below within <strong>1 hour</strong>:</p>
      <p style="text-align:center">
        <a href="{reset_link}"
           style="background:#2563eb;color:#fff;padding:12px 24px;border-radius:6px;text-decoration:none;display:inline-block">
          Reset Password
        </a>
      </p>
      <p style="color:#6b7280;font-size:13px">If you did not request this, ignore this email.</p>
      <hr/>
      <p style="color:#6b7280;font-size:12px">VisionDx · AI-powered malaria diagnostics</p>
    </div>
    """
    await _send(to_email, "VisionDx – Reset your password", html)


async def send_verification_email(to_email: str, full_name: str, verify_link: str) -> None:
    html = f"""
    <div style="font-family:sans-serif;max-width:560px;margin:auto">
      <h2>Welcome to VisionDx, {full_name}!</h2>
      <p>Please verify your email address to activate your account:</p>
      <p style="text-align:center">
        <a href="{verify_link}"
           style="background:#16a34a;color:#fff;padding:12px 24px;border-radius:6px;text-decoration:none;display:inline-block">
          Verify Email
        </a>
      </p>
      <p style="color:#6b7280;font-size:13px">This link expires in <strong>24 hours</strong>.</p>
      <hr/>
      <p style="color:#6b7280;font-size:12px">VisionDx · AI-powered malaria diagnostics</p>
    </div>
    """
    await _send(to_email, "VisionDx – Verify your email", html)
