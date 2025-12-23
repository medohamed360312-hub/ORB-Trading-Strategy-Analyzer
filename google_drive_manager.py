import json
import base64
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

class GoogleDriveManager:
    def __init__(self):
        self.service = None
        self.authenticate()
    
    def authenticate(self):
        """Use Service Account instead of OAuth2.0 flow"""
        try:
            creds = None
            
            # Try base64 from environment first (for GitHub Actions)
            creds_json_b64 = os.getenv("GOOGLE_DRIVE_CREDENTIALS")
            if creds_json_b64:
                creds_json = base64.b64decode(creds_json_b64).decode("utf-8")
                creds_dict = json.loads(creds_json)
                creds = service_account.Credentials.from_service_account_info(
                    creds_dict, scopes=SCOPES
                )
            
            # Try local file
            elif os.path.exists("service_account.json"):
                creds = service_account.Credentials.from_service_account_file(
                    "service_account.json", scopes=SCOPES
                )
            
            else:
                print("❌ No credentials found. Set GOOGLE_DRIVE_CREDENTIALS env var or add service_account.json")
                return False
            
            self.service = build("drive", "v3", credentials=creds)
            print("✅ Authenticated with Google Drive (Service Account)")
            return True
            
        except Exception as e:
            print(f"❌ GDrive auth failed: {str(e)[:100]}")
            return False
    
    def upload_csv(self, local_file_path, drive_file_name, folder_id=None):
        """Upload or update a CSV file on Google Drive."""
        try:
            if not self.service:
                print("❌ Not authenticated with Google Drive.")
                return False
            
            # Search for file
            query = f"name='{drive_file_name}' and trashed=false"
            if folder_id:
                query += f" and parents='{folder_id}'"
            
            results = self.service.files().list(
                q=query,
                spaces="drive",
                fields="files(id)",
                pageSize=1
            ).execute()
            
            file_id = results["files"][0]["id"] if results.get("files") else None
            file_metadata = {"name": drive_file_name}
            media = MediaFileUpload(local_file_path, mimetype="text/csv", resumable=True)
            
            if file_id:
                self.service.files().update(
                    fileId=file_id,
                    body=file_metadata,
                    media_body=media,
                    fields="id"
                ).execute()
            else:
                body = {"name": drive_file_name}
                if folder_id:
                    body["parents"] = [folder_id]
                self.service.files().create(
                    body=body,
                    media_body=media,
                    fields="id"
                ).execute()
            
            print(f"✅ GDrive upload successful: {drive_file_name}")
            return True
            
        except Exception as e:
            print(f"❌ GDrive upload failed: {str(e)[:100]}")
            return False
    
    def download_csv(self, drive_file_name, local_file_path):
        """Download CSV from Google Drive."""
        try:
            if not self.service:
                print("❌ Not authenticated with Google Drive.")
                return False
            
            results = self.service.files().list(
                q=f"name='{drive_file_name}' and trashed=false",
                spaces="drive",
                fields="files(id)",
                pageSize=1
            ).execute()
            
            if not results.get("files"):
                print(f"❌ GDrive file not found: {drive_file_name}")
                return False
            
            file_id = results["files"][0]["id"]
            request = self.service.files().get_media(fileId=file_id)
            
            with open(local_file_path, "wb") as f:
                f.write(request.execute())
            
            print(f"✅ GDrive download successful: {drive_file_name}")
            return True
            
        except Exception as e:
            print(f"❌ GDrive download failed: {str(e)[:100]}")
            return False
