import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth
import json

# Initialize Firebase Admin SDK
cred = credentials.Certificate('build-with-ai-sri-lanka-firebase-adminsdk-fbsvc-b857c2e764.json')
firebase_admin.initialize_app(cred)

# Read the CSV file
df = pd.read_csv('bwai_reg_updated.csv')

# Convert registration_date to datetime for proper sorting
df['registration_date'] = pd.to_datetime(df['registration_date'])

# Sort by registration_date and keep the last record for each email
df = df.sort_values('registration_date').groupby('email').last().reset_index()

def update_user_info(email):
    try:
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
print(f"Total unique records after removing duplicates: {len(df)}")
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