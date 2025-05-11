import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth
import json
import re

def clean_email(email):
    if pd.isna(email):
        return None
    # Convert to string and lowercase
    email = str(email).lower().strip()
    # Fix common issues
    if not '@' in email:
        if 'gmail.com' in email:
            email = email.replace('gmail.com', '@gmail.com')
        elif 'yahoo.com' in email:
            email = email.replace('yahoo.com', '@yahoo.com')
    return email

# Initialize Firebase Admin SDK
cred = credentials.Certificate('build-with-ai-sri-lanka-firebase-adminsdk-fbsvc-b857c2e764.json')
firebase_admin.initialize_app(cred)

# Read the CSV files
df_reg = pd.read_csv('bwai_reg.csv')
df_rsvp = pd.read_csv('bwai_rsvp.csv')

# Clean email addresses
df_reg['email'] = df_reg['email'].apply(clean_email)
df_rsvp['email'] = df_rsvp['email'].apply(clean_email)

# Remove rows with invalid emails
df_reg = df_reg.dropna(subset=['email'])
df_rsvp = df_rsvp.dropna(subset=['email'])

# Convert registration_date to datetime for proper sorting
df_reg['registration_date'] = pd.to_datetime(df_reg['registration_date'])

# Sort by registration_date and keep the last record for each email in registration data
df_reg = df_reg.sort_values('registration_date').groupby('email').last().reset_index()

# Merge data, giving priority to RSVP data
df = pd.concat([df_reg, df_rsvp], ignore_index=True)
df = df.sort_values('email').groupby('email').last().reset_index()

print(f"Total records from registration: {len(df_reg)}")
print(f"Total records from RSVP: {len(df_rsvp)}")
print(f"Total unique records after merge: {len(df)}")

def update_user_info(email):
    try:
        if not isinstance(email, str) or not '@' in email:
            return None, None
        # Get user by email
        user = auth.get_user_by_email(email)
        return user.display_name, user.photo_url
    except auth.UserNotFoundError:
        return None, None
    except Exception as e:
        print(f"Error fetching user data for {email}: {str(e)}")
        return None, None

# Update name and picture fields
print("Starting to update user information...")
updated_count = 0

for index, row in df.iterrows():
    email = row['email']
    name, picture = update_user_info(email)
    
    if name or picture:
        if name:
            df.at[index, 'name'] = name
        if picture:
            df.at[index, 'picture'] = picture
        updated_count += 1
        
        if updated_count % 10 == 0:
            print(f"Updated {updated_count} records...")

print(f"Finished updating. Total records updated: {updated_count}")

# Save the updated CSV
df.to_csv('bwai_reg_updated.csv', index=False)
print("Updated CSV saved as 'bwai_reg_updated.csv'")