from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import bcrypt
import firebase_admin
import streamlit as st
from firebase_admin import credentials, firestore
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

LOCAL_DATA_DIR = Path(".streamlit")
LOCAL_USERS_FILE = LOCAL_DATA_DIR / "local_users.json"


def _secret_section(name: str) -> dict[str, Any]:
    try:
        value = st.secrets[name]
    except Exception:
        return {}
    return dict(value)


def _firebase_credentials() -> dict[str, Any]:
    firebase_secrets = _secret_section("firebase")
    if not firebase_secrets:
        return {}

    private_key = firebase_secrets.get("private_key", "")
    return {
        "type": firebase_secrets.get("type"),
        "project_id": firebase_secrets.get("project_id"),
        "private_key_id": firebase_secrets.get("private_key_id"),
        "private_key": private_key.replace("\\n", "\n"),
        "client_email": firebase_secrets.get("client_email"),
        "client_id": firebase_secrets.get("client_id"),
        "auth_uri": firebase_secrets.get("auth_uri"),
        "token_uri": firebase_secrets.get("token_uri"),
        "auth_provider_x509_cert_url": firebase_secrets.get("auth_provider_x509_cert_url"),
        "client_x509_cert_url": firebase_secrets.get("client_x509_cert_url"),
    }


def init_firebase() -> bool:
    if firebase_admin._apps:
        return True

    credentials_payload = _firebase_credentials()
    if not credentials_payload:
        return False

    try:
        cred = credentials.Certificate(credentials_payload)
        firebase_admin.initialize_app(cred)
        return True
    except Exception:
        return False


def _load_local_users() -> dict[str, Any]:
    if not LOCAL_USERS_FILE.exists():
        return {}

    try:
        return json.loads(LOCAL_USERS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_local_users(users: dict[str, Any]) -> None:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def create_drive_service():
    credentials_info = _secret_section("gcp_service_account")
    if not credentials_info:
        return None

    try:
        creds = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=["https://www.googleapis.com/auth/drive.file"],
        )
        return build("drive", "v3", credentials=creds)
    except Exception:
        return None


def upload_file_to_drive(service, file_path: str | Path, file_name: str, mime_type: str, folder_id: str):
    if service is None:
        return None

    path = Path(file_path)
    if not path.exists():
        return None

    file_metadata = {
        "name": file_name,
        "mimeType": mime_type,
        "parents": [folder_id],
    }
    media = MediaFileUpload(str(path), mimetype=mime_type)
    try:
        uploaded_file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
    except (HttpError, Exception):
        return None

    return uploaded_file.get("id")


def get_firestore_client():
    if not init_firebase():
        return None

    try:
        return firestore.client()
    except Exception:
        return None


def add_user_to_firestore(username, data):
    db = get_firestore_client()
    if db is not None:
        db.collection("users").document(username).set(data)
        return f"User '{username}' data added/updated in Firestore."

    users = _load_local_users()
    users[username] = data
    _save_local_users(users)
    return f"User '{username}' data added/updated locally."


def fetch_user_from_firestore(username):
    db = get_firestore_client()
    if db is not None:
        user_doc = db.collection("users").document(username).get()
        return user_doc.to_dict() if user_doc.exists else None

    return _load_local_users().get(username)


def hash_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed_password.decode("utf-8")


def check_password(hashed_password, user_password):
    return bcrypt.checkpw(
        user_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )
