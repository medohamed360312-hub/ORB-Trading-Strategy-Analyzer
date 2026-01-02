import json
import base64
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

# SCOPES for Google Sheets API
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets"
]

class GoogleDriveManager:
    def __init__(self):
        self.sheets_service = None
        self.drive_service = None
        self.authenticate()

    def authenticate(self):
        """Authenticate with Google APIs"""
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
                print("❌ No credentials found")
                return False

            self.drive_service = build("drive", "v3", credentials=creds)
            self.sheets_service = build("sheets", "v4", credentials=creds)
            print("✅ Authenticated with Google Drive & Sheets API")
            return True

        except Exception as e:
            print(f"❌ Auth failed: {str(e)[:100]}")
            return False

    def create_or_get_sheet(self, sheet_name):
        """Create Google Sheet or get existing one by name"""
        try:
            if not self.drive_service:
                print("❌ Not authenticated")
                return None

            # Search for existing sheet
            query = f"name='{sheet_name}' and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
            results = self.drive_service.files().list(
                q=query,
                spaces="drive",
                fields="files(id)",
                pageSize=1
            ).execute()

            # If exists, return ID
            if results.get("files"):
                sheet_id = results["files"][0]["id"]
                print(f"✅ Found existing Google Sheet: {sheet_name}")
                return sheet_id

            # If not, create new
            file_metadata = {
                "name": sheet_name,
                "mimeType": "application/vnd.google-apps.spreadsheet"
            }

            file = self.drive_service.files().create(
                body=file_metadata,
                fields="id"
            ).execute()

            sheet_id = file.get("id")
            print(f"✅ Created new Google Sheet: {sheet_name}")
            return sheet_id

        except Exception as e:
            print(f"❌ Sheet creation failed: {str(e)[:100]}")
            return None

    def upload_headers(self, spreadsheet_id, headers):
        """Upload header row to Google Sheet"""
        try:
            if not self.sheets_service:
                return False

            body = {"values": [headers]}
            self.sheets_service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range="Sheet1!A1",
                valueInputOption="USER_ENTERED",
                body=body
            ).execute()

            print(f"✅ Headers uploaded to Google Sheet")
            return True

        except Exception as e:
            print(f"❌ Header upload failed: {str(e)[:100]}")
            return False

    def append_row(self, spreadsheet_id, row_data):
        """Append a single row to Google Sheet"""
        try:
            if not self.sheets_service:
                return False

            body = {"values": [row_data]}
            self.sheets_service.spreadsheets().values().append(
                spreadsheetId=spreadsheet_id,
                range="Sheet1!A:A",
                valueInputOption="USER_ENTERED",
                body=body
            ).execute()

            return True

        except Exception as e:
            print(f"❌ Row append failed: {str(e)[:100]}")
            return False

    def get_sheet_url(self, spreadsheet_id):
        """Get the URL of the Google Sheet"""
        try:
            if not self.sheets_service:
                return None

            sheet = self.sheets_service.spreadsheets().get(
                spreadsheetId=spreadsheet_id,
                fields="spreadsheetUrl"
            ).execute()
            return sheet.get("spreadsheetUrl")

        except Exception as e:
            print(f"❌ Failed to get URL: {str(e)[:100]}")
            return None
