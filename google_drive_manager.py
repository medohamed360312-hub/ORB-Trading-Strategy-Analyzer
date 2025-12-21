#!/usr/bin/env python3
"""
GOOGLE DRIVE INTEGRATION MODULE FOR FOREX ORB STRATEGY
Handles uploading and managing CSV files on Google Drive
"""

import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pandas as pd
from io import StringIO

# Google Drive API Scopes
SCOPES = ['https://www.googleapis.com/auth/drive.file']

class GoogleDriveManager:
    def __init__(self, credentials_file='credentials.json', token_file='token.pickle'):
        """
        Initialize Google Drive Manager
        
        Args:
            credentials_file: Path to credentials.json (from Google Cloud Console)
            token_file: Path to token.pickle (auto-generated after first auth)
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with Google Drive API"""
        creds = None
        
        # Load existing token if available
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, request new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(
                        f"❌ {self.credentials_file} not found!\n"
                        "Please download it from Google Cloud Console:\n"
                        "1. Go to https://console.cloud.google.com\n"
                        "2. Create OAuth 2.0 credentials (Desktop app)\n"
                        "3. Save as credentials.json"
                    )
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save token for future use
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('drive', 'v3', credentials=creds)
        print("✅ Google Drive authenticated successfully")
    
    def find_file(self, filename):
        """Find file on Google Drive by name"""
        try:
            query = f"name='{filename}' and trashed=false"
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, webViewLink)',
                pageSize=1
            ).execute()
            
            files = results.get('files', [])
            if files:
                return files[0]
            return None
        except Exception as e:
            print(f"❌ Error finding file: {str(e)}")
            return None
    
    def upload_csv(self, local_file, drive_filename=None):
        """
        Upload CSV file to Google Drive
        
        Args:
            local_file: Path to local CSV file
            drive_filename: Name to use on Google Drive (default: same as local)
        
        Returns:
            file_id or None if failed
        """
        if drive_filename is None:
            drive_filename = os.path.basename(local_file)
        
        try:
            # Check if file exists
            existing_file = self.find_file(drive_filename)
            
            file_metadata = {'name': drive_filename}
            media = MediaFileUpload(local_file, mimetype='text/csv')
            
            if existing_file:
                # Update existing file
                file = self.service.files().update(
                    fileId=existing_file['id'],
                    media_body=media,
                    fields='id, webViewLink'
                ).execute()
                print(f"✅ Updated file on Google Drive: {drive_filename}")
                print(f"   Link: {file.get('webViewLink', 'N/A')}")
            else:
                # Create new file
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id, webViewLink'
                ).execute()
                print(f"✅ Created file on Google Drive: {drive_filename}")
                print(f"   Link: {file.get('webViewLink', 'N/A')}")
            
            return file.get('id')
        
        except Exception as e:
            print(f"❌ Error uploading file: {str(e)}")
            return None
    
    def download_csv(self, drive_filename):
        """
        Download CSV file from Google Drive as DataFrame
        
        Args:
            drive_filename: Name of file on Google Drive
        
        Returns:
            DataFrame or None if failed
        """
        try:
            file_info = self.find_file(drive_filename)
            if not file_info:
                print(f"❌ File not found on Google Drive: {drive_filename}")
                return None
            
            # Download file content
            request = self.service.files().get_media(fileId=file_info['id'])
            content = request.execute()
            
            # Convert to DataFrame
            df = pd.read_csv(StringIO(content.decode('utf-8')))
            print(f"✅ Downloaded file from Google Drive: {drive_filename}")
            return df
        
        except Exception as e:
            print(f"❌ Error downloading file: {str(e)}")
            return None


def prepend_to_csv(local_file, new_rows_df, drive_filename=None):
    """
    Add new rows to the TOP of CSV file (newest first)
    
    Args:
        local_file: Path to local CSV file
        new_rows_df: DataFrame with new rows to add
        drive_filename: Optional Google Drive filename
    
    Returns:
        True if successful
    """
    try:
        # Read existing CSV
        if os.path.exists(local_file):
            existing_df = pd.read_csv(local_file)
        else:
            existing_df = pd.DataFrame()
        
        # Combine: new rows on TOP
        if not existing_df.empty:
            combined_df = pd.concat([new_rows_df, existing_df], ignore_index=True)
        else:
            combined_df = new_rows_df
        
        # Write back to local file
        combined_df.to_csv(local_file, index=False)
        print(f"✅ Updated local CSV (newest entries first): {local_file}")
        
        # Also update Google Drive if filename provided
        if drive_filename:
            try:
                drive_manager = GoogleDriveManager()
                drive_manager.upload_csv(local_file, drive_filename)
            except Exception as e:
                print(f"⚠️  Could not update Google Drive: {str(e)}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error prepending to CSV: {str(e)}")
        return False
