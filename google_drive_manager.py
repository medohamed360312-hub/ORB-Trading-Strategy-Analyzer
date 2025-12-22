"""
Google Drive Manager - simplified version
Uses OAuth 2.0 with credentials.json or GOOGLE_DRIVE_CREDENTIALS (base64)
"""

import os
import pickle
import base64
import json

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


class GoogleDriveManager:
    def __init__(self):
        self.service = None
        self.authenticate()

    def authenticate(self):
        """Authenticate with Google Drive and create token.pickle if needed."""
        try:
            creds = None

            # 1) Try existing token.pickle
            if os.path.exists("token.pickle"):
                with open("token.pickle", "rb") as token:
                    creds = pickle.load(token)

            # 2) If no token, use credentials.json or GOOGLE_DRIVE_CREDENTIALS
            if not creds:
                # Try base64 from env first
                creds_json_b64 = os.getenv("GOOGLE_DRIVE_CREDENTIALS")

                if creds_json_b64:
                    creds_json = base64.b64decode(creds_json_b64).decode("utf-8")
                    creds_dict = json.loads(creds_json)
                    flow = InstalledAppFlow.from_client_config(creds_dict, SCOPES)
                elif os.path.exists("credentials.json"):
                    flow = InstalledAppFlow.from_client_secrets_file(
                        "credentials.json", SCOPES
                    )
                else:
                    print("❌ No credentials.json or GOOGLE_DRIVE_CREDENTIALS found.")
                    return False

                creds = flow.run_local_server(port=0)

                # Save token for next runs
                with open("token.pickle", "wb") as token:
                    pickle.dump(creds, token)

            self.service = build("drive", "v3", credentials=creds)
            return True

        except Exception as e:
            print(f"❌ GDrive auth failed: {str(e)[:100]}")
            return False

    def upload_csv(self, local_file_path, drive_file_name):
        """Upload or update a CSV file on Google Drive."""
        try:
            if not self.service:
                print("❌ Not authenticated with Google Drive.")
                return False

            # Check if file already exists
            results = (
                self.service.files()
                .list(
                    q=f"name='{drive_file_name}' and trashed=false",
                    spaces="drive",
                    fields="files(id)",
                    pageSize=1,
                )
                .execute()
            )
            file_id = results["files"][0]["id"] if results.get("files") else None

            file_metadata = {"name": drive_file_name}
            media = MediaFileUpload(local_file_path, mimetype="text/csv", resumable=True)

            if file_id:
                self.service.files().update(
                    fileId=file_id,
                    body=file_metadata,
                    media_body=media,
                    fields="id",
                ).execute()
            else:
                self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields="id",
                ).execute()

            return True

        except Exception as e:
            print(f"❌ GDrive upload failed: {str(e)[:100]}")
            return False

    def download_csv(self, drive_file_name, local_file_path):
        """Download CSV from Google Drive (if exists)."""
        try:
            if not self.service:
                print("❌ Not authenticated with Google Drive.")
                return False

            results = (
                self.service.files()
                .list(
                    q=f"name='{drive_file_name}' and trashed=false",
                    spaces="drive",
                    fields="files(id)",
                    pageSize=1,
                )
                .execute()
            )
            if not results.get("files"):
                print(f"❌ GDrive file not found: {drive_file_name}")
                return False

            file_id = results["files"][0]["id"]
            request = self.service.files().get_media(fileId=file_id)
            with open(local_file_path, "wb") as f:
                f.write(request.execute())

            return True

        except Exception as e:
            print(f"❌ GDrive download failed: {str(e)[:100]}")
            return False
