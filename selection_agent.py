import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import os
import re
import asyncio
from typing import List, Dict
import math
from concurrent.futures import ThreadPoolExecutor
import csv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from google.adk.tools import FunctionTool, agent_tool
from google.adk.agents import LlmAgent 
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from google.genai import Client

from pydantic import BaseModel, Field

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GITHUB_PAT = os.environ.get("GITHUB_PAT")

# Initialize Google AI client with API key
if GOOGLE_API_KEY:
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY  # Ensure it's in environment
    google_client = Client(api_key=GOOGLE_API_KEY)
else:
    google_client = None

INPUT_CSV_PATH = "devfest-codelabs-ai-&-ml-all-statuses-2025-12-27.csv"
OUTPUT_CSV_PATH = "bwai_graded.csv"
APP_NAME = "ParticipantScoringApp"

# Batch processing configuration
BATCH_SIZE = 10  # Number of participants to process in each batch
MAX_CONCURRENT_TASKS = 10  # Maximum number of concurrent tasks
SEMAPHORE_LIMIT = 10  # Maximum number of simultaneous API calls

# --- Helper: Score Parsing from LLM --- (No change)
def parse_llm_score(score_str: str, max_score: int, field_name: str, default_score: int = 0) -> int:
    if not score_str: return default_score
    try:
        match = re.search(r"score.*?(\d+)|(\d+)", str(score_str).lower())
        if match:
            num_str = match.group(1) or match.group(2)
            score = int(num_str)
            if 0 <= score <= max_score:
                return score
            return default_score
        else:
            cleaned_score_str = "".join(filter(str.isdigit, str(score_str)))
            if cleaned_score_str:
                score = int(cleaned_score_str)
                if 0 <= score <= max_score:
                    return score
            return default_score
    except ValueError: return default_score
    except Exception: return default_score

# --- Rule-Based Scoring Functions ---
def _score_github_contributions_rule_based(contributions: int) -> int:
    if contributions >= 1000: return 50
    if contributions > 500: return 25
    if contributions > 200: return 15
    if contributions > 50:  return 10
    if contributions >= 10: return 5
    return 0

def _score_status_type_rule_based(status_text: str) -> int:
    """
    Rule-based scoring for status type (student or professional).
    - Professional: 30 points
    - Student: 10 points
    - Other/Unknown: 0 points
    """
    if not status_text:
        return 0
    
    status_lower = status_text.lower().strip()
    
    # Check for professional keywords
    if "professional" in status_lower:
        return 30
    # Check for student keywords
    elif "student" in status_lower:
        return 10
    else:
        return 0

# --- Tool Function Implementations ---
def status_type_scoring_tool_func(status_type: str) -> dict:
    """
    Rule-based status type scoring tool. Returns a score based on whether the status is professional or student.
    Output keys: "score".
    """
    score = _score_status_type_rule_based(status_type)
    return {"score": score}

def github_processing_tool_func(github_profile_url: str) -> dict:
    """
    Fetches GitHub contributions for a given profile URL and returns a score based on contribution count.
    Also returns username, contribution count, and any fetch error.
    Output keys: "github_score", "github_username", "github_contributions", "github_fetch_error".
    """
    username_extracted = ""; contributions_count = 0; error_msg = None; score = 0
    if not github_profile_url or not isinstance(github_profile_url, str): error_msg = "Invalid GitHub URL provided."
    else:
        match = re.search(r"github\.com/([^/]+)", github_profile_url)
        if not match: error_msg = "Could not extract username from URL."
        else:
            username_extracted = match.group(1)
            if not GITHUB_PAT: error_msg = "GitHub PAT not configured. Cannot fetch contributions."
            else:
                headers = {"Authorization": f"bearer {GITHUB_PAT}", "Content-Type": "application/json"}; to_date = datetime.utcnow(); from_date = to_date - timedelta(days=365)
                query_str = """query($username: String!, $fromDate: DateTime!, $toDate: DateTime!) { user(login: $username) { contributionsCollection(from: $fromDate, to: $toDate) { contributionCalendar { totalContributions } } } }"""
                variables = {"username": username_extracted, "fromDate": from_date.isoformat() + "Z", "toDate": to_date.isoformat() + "Z"}
                try:
                    response = requests.post("https://api.github.com/graphql", headers=headers, json={"query": query_str, "variables": variables}, timeout=15)
                    response.raise_for_status(); data = response.json()
                    if "errors" in data: error_msg = f"GitHub API error: {data['errors']}"
                    else: contributions_count = data.get("data", {}).get("user", {}).get("contributionsCollection", {}).get("contributionCalendar", {}).get("totalContributions", 0)
                except requests.exceptions.RequestException as e: error_msg = f"GitHub fetch HTTP error: {str(e)}"
                except Exception as e: error_msg = f"Unexpected error during GitHub fetch: {str(e)}"
    if not error_msg: score = _score_github_contributions_rule_based(contributions_count)
    return {"github_score": score, "github_username": username_extracted if username_extracted else None, "github_contributions": contributions_count, "github_fetch_error": error_msg}

# --- Specialist Agent Input/Output Schemas (Pydantic models for agent_tool wrappers) ---
class SingleTextEvalInputArgs(BaseModel):
    text_to_evaluate: str

class ScoreOutput(BaseModel):
    score: int = Field(description="The calculated integer score for the criterion.")

# --- Specialist LLM Agent Prompts (Content no change) ---
PROMPT_Q5_EXPECTATIONS = """You are an AI evaluating participant expectations for an AI & ML codelab.
The Q5 response is in 'text_to_evaluate' argument.
Codelab sessions cover: AI agents, RAG, function calling, agentic AI, fine-tuning, embeddings, and practical ML/AI implementation.
Score based on alignment and enthusiasm for hands-on technical learning:
- 10-15 pts: Strong alignment with specific technical topics (AI agents, RAG, function calling). Clear desire to build/apply. Specific technical goals mentioned.
- 5-9 pts: General AI/ML interest. Upskilling goals. Proactive but not tied to specific technical topics.
- 1-4 pts: Vague interest. Generic learning/networking. Not closely related to technical depth.
- 0 pts: No relevant information.
Return ONLY a JSON object: {{"score": <integer_score>}}.
"""
PROMPT_Q1 = """Evaluate understanding of AI hallucination from the response provided in 'text_to_evaluate' argument.
The question asks: What is the error called when an AI confidently provides wrong information?
Score based on accuracy and depth:
- 10 pts: Correctly identifies hallucination with clear explanation of why it occurs (training patterns, prediction, etc.).
- 6-8 pts: Correctly identifies hallucination with some explanation.
- 3-5 pts: Mentions hallucination or related concept but incomplete understanding.
- 1-2 pts: Vague or partially correct answer.
- 0 pts: Incorrect or no answer.
Return ONLY a JSON object: {{"score": <integer_score>}}.
"""
PROMPT_Q2 = """Evaluate understanding of multi-agent hallucination prevention from the response in 'text_to_evaluate' argument.
The question asks: How do you prevent hallucination when passing info between AI agents?
Score based on accuracy and technical depth:
- 8-10 pts: Mentions structured state/shared state/JSON, validation, grounding, or other correct technical approaches with clear explanation.
- 5-7 pts: Mentions correct concepts (shared memory, structured data) but less detailed.
- 2-4 pts: Vague answer (e.g., "validate", "check") without specific mechanism.
- 0-1 pts: Incorrect or no answer.
Return ONLY a JSON object: {{"score": <integer_score>}}.
"""
PROMPT_Q3 = """Evaluate understanding of function calling in LLMs from the response in 'text_to_evaluate' argument.
The question asks: How does function calling work in LLMs?
Score based on technical understanding:
- 8-10 pts: Explains that LLM outputs structured response/JSON to trigger external tools, mentions training patterns, token prediction, or schema matching.
- 5-7 pts: Mentions calling external tools/APIs correctly but less technical detail.
- 2-4 pts: Vague understanding (e.g., "it calls functions" without explaining how).
- 0-1 pts: Incorrect or no answer.
Return ONLY a JSON object: {{"score": <integer_score>}}.
"""

PROMPT_Q4 = """Evaluate understanding of RAG vs Fine-tuning from the response in 'text_to_evaluate' argument.
The question asks: Why is RAG preferred over fine-tuning for private PDFs and SQL databases?
Score based on technical understanding:
- 8-10 pts: Correctly explains RAG retrieves real-time data, avoids retraining, keeps data private, handles changing data, reduces hallucinations.
- 5-7 pts: Mentions some correct reasons (real-time data, privacy) but less complete.
- 2-4 pts: Vague understanding or partial correctness.
- 0-1 pts: Incorrect or no answer.
Return ONLY a JSON object: {{"score": <integer_score>}}.
"""

# --- Orchestrator Agent Prompt Template (No change in content) ---
ORCHESTRATOR_PROMPT_TEMPLATE = """
You are a master evaluator for AI conference participants. For the given participant data, your goal is to determine all individual scores by calling specialized scoring agents/tools, then calculate the total score and star rating.

**Participant Data:**
Name: {name}
Email: {email}
Status Type: {status_type}
GitHub Profile URL: {github_profile_url}
Q1 Response (AI Hallucination): {q1_response}
Q2 Response (Multi-agent Hallucination Prevention): {q2_response}
Q3 Response (Function Calling): {q3_response}
Q4 Response (RAG vs Fine-tuning): {q4_response}
Q5 Response (Expectations): {q5_response}

**Evaluation Steps & Tool Usage:**
1.  Call the `status_type_scoring_tool_func` tool. Provide the argument `status_type` set to the participant's `status_type` text. Expect JSON `{{ "score": <num> }}`. Let this be `status_score_json`.
2.  Call the `github_processing_tool_func` tool. Provide the argument `github_profile_url` set to participant's `github_profile_url`. Expect JSON `{{ "github_score": <num>, "github_username": <str_or_null>, "github_contributions": <num>, "github_fetch_error": <str_or_null> }}`. Let this be `github_details_json`.
3.  Call the `Q1ScoringAgent` tool. Provide argument `text_to_evaluate` set to `q1_response`. Expect JSON `{{ "score": <num> }}`. Let this be `q1_score_json`.
4.  Call the `Q2ScoringAgent` tool. Provide argument `text_to_evaluate` set to `q2_response`. Expect JSON `{{ "score": <num> }}`. Let this be `q2_score_json`.
5.  Call the `Q3ScoringAgent` tool. Provide argument `text_to_evaluate` set to `q3_response`. Expect JSON `{{ "score": <num> }}`. Let this be `q3_score_json`.
6.  Call the `Q4ScoringAgent` tool. Provide argument `text_to_evaluate` set to `q4_response`. Expect JSON `{{ "score": <num> }}`. Let this be `q4_score_json`.
7.  Call the `Q5ScoringAgent` tool. Provide argument `text_to_evaluate` set to `q5_response`. Expect JSON `{{ "score": <num> }}`. Let this be `q5_score_json`.

**Data Aggregation and Final Calculation:**
* `status_score` = `status_score_json['score']`
* `github_score` = `github_details_json['github_score']` (also get `github_username`, `github_contributions`, `github_fetch_error` from `github_details_json`)
* `q1_score` = `q1_score_json['score']`
* `q2_score` = `q2_score_json['score']`
* `q3_score` = `q3_score_json['score']`
* `q4_score` = `q4_score_json['score']`
* `q5_score` = `q5_score_json['score']`
* Sum all these scores to get `total_score`.
* Calculate an integer `star_rating` (0-5) based on `total_score`: 90-100=5, 75-89=4, 60-74=3, 40-59=2, 1-39=1, 0=0.

**Output Format:**
You MUST return ONLY a single valid JSON object with ALL the following keys:
"github_username", "github_contributions", "github_score", "status_score", "q1_score", "q2_score", "q3_score", "q4_score", "q5_score", "total_score", "star_rating", "github_fetch_error".
Ensure all score values and star_rating are integers.

IMPORTANT: DO NOT output any Python code, variable assignments, or calculations. ONLY output a raw JSON object like this example (replace with actual values):
{{
  "github_username": "username",
  "github_contributions": 157,
  "github_score": 10,
  "status_score": 10,
  "q1_score": 8,
  "q2_score": 7,
  "q3_score": 9,
  "q4_score": 8,
  "q5_score": 12,
  "total_score": 64,
  "star_rating": 3,
  "github_fetch_error": null
}}
"""

def append_single_result_to_csv(participant_data: pd.Series, result: dict, output_path: str):
    """
    Append a single participant result to CSV immediately after processing.
    """
    # Merge participant data with scoring results
    merged_result = participant_data.to_dict()
    if isinstance(result, dict) and "error" not in result:
        # Filter out error-related fields
        result = {k: v for k, v in result.items() if k not in ['error', 'raw_content', 'github_fetch_error']}
        merged_result.update(result)
    
    # Convert to DataFrame for easier CSV handling
    result_df = pd.DataFrame([merged_result])
    
    # Handle numeric columns
    score_cols_to_numeric = [
        "github_contributions", "github_score", "status_score",
        "q1_score", "q2_score", "q3_score", "q4_score", "q5_score",
        "total_score", "star_rating"
    ]
    
    for col in score_cols_to_numeric:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0).astype(int)
        else:
            result_df[col] = 0
    
    try:
        # Write to CSV with proper escaping and quoting
        result_df.to_csv(
            output_path, 
            mode='a',
            header=not os.path.exists(output_path),
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            escapechar='\\',
            doublequote=True
        )
    except Exception as e:
        print(f"Error appending result for {participant_data.get('Email', 'N/A')} to CSV: {e}")

async def process_single_participant_multi_agent(
    participant_data: pd.Series,
    runner: Runner, 
    session_service: InMemorySessionService, 
    app_name: str,
    participant_index: int,
    output_path: str = None
) -> dict:
    # (No change in this function's internal logic)
    prompt = ORCHESTRATOR_PROMPT_TEMPLATE.format(
        name=participant_data.get('Name', 'N/A'), email=participant_data.get('Email', 'N/A'),
        status_type=str(participant_data.get('Status Type', '')),
        github_profile_url=str(participant_data.get('GitHub', '')),
        q1_response=str(participant_data.get('Q1', '')),
        q2_response=str(participant_data.get('Q2', '')),
        q3_response=str(participant_data.get('Q3', '')),
        q4_response=str(participant_data.get('Q4', '')),
        q5_response=str(participant_data.get('Q5', ''))
    )
    user_id = f"participant_idx_{participant_index}" # Make user_id more distinct from email for testing
    session_id = f"session_idx_{participant_index}"
    session = session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
    if not session: session = session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    content = genai_types.Content(role='user', parts=[genai_types.Part(text=prompt)])
    final_response_content = "";
    try:
        loop = asyncio.get_event_loop()
        events_iterable = await loop.run_in_executor(None, lambda: runner.run(user_id=user_id, session_id=session_id, new_message=content))
        for event in events_iterable:
            if event.author == runner.agent.name and event.is_final_response():
                if hasattr(event.content, 'parts') and event.content.parts: final_response_content = event.content.parts[0].text
                elif hasattr(event.content, 'text'): final_response_content = event.content.text
                break
        if final_response_content:
            # Clean up markdown code blocks if present
            if final_response_content.startswith("```"):
                # Handle both ```json and ``` formats
                if final_response_content.startswith("```json"):
                    final_response_content = final_response_content[7:]
                elif final_response_content.startswith("```python"):
                    final_response_content = final_response_content[10:]
                else:
                    final_response_content = final_response_content[3:]
            if final_response_content.endswith("```"): 
                final_response_content = final_response_content[:-3]
                
            # Try to extract JSON from the content
            # First look for a JSON object pattern
            import re
            json_pattern = r'\{[\s\S]*\}'
            json_match = re.search(json_pattern, final_response_content)
            
            if json_match:
                try:
                    response_json = json.loads(json_match.group(0))
                    # Convert star_rating to int if it's a string
                    if 'star_rating' in response_json and isinstance(response_json['star_rating'], str):
                        try: response_json['star_rating'] = int(response_json['star_rating'])
                        except ValueError: response_json['star_rating'] = 0
                    if output_path:
                        append_single_result_to_csv(participant_data, response_json, output_path)
                    return response_json
                except json.JSONDecodeError:
                    # If we can't parse the extracted JSON, fall through to the next attempt
                    pass
            
            # Try parsing the entire content as JSON
            try:
                response_json = json.loads(final_response_content.strip())
                if 'star_rating' in response_json and isinstance(response_json['star_rating'], str):
                    try: response_json['star_rating'] = int(response_json['star_rating'])
                    except ValueError: response_json['star_rating'] = 0
                if output_path:
                    append_single_result_to_csv(participant_data, response_json, output_path)
                return response_json
            except json.JSONDecodeError:
                # If we still can't parse it, try to extract the print statement output if it's Python code
                print_pattern = r'print\(f[\"\'](\{.*\})[\"\'](\))'
                print_match = re.search(print_pattern, final_response_content)
                if print_match:
                    try:
                        # Reconstruct a JSON object from the print statement
                        print_content = print_match.group(1)
                        # Replace any Python None with null for JSON compatibility
                        print_content = print_content.replace('None', 'null')
                        response_json = json.loads(print_content)
                        if 'star_rating' in response_json and isinstance(response_json['star_rating'], str):
                            try: response_json['star_rating'] = int(response_json['star_rating'])
                            except ValueError: response_json['star_rating'] = 0
                        if output_path:
                            append_single_result_to_csv(participant_data, response_json, output_path)
                        return response_json
                    except (json.JSONDecodeError, IndexError):
                        # If we can't extract from print statement, give up and return error
                        pass
                
                # If all parsing attempts fail, return the error with the raw content
                print(f"Error: Could not parse JSON from Orchestrator response for {participant_data.get('name', 'N/A')}")
                result = {"error": "JSONDecodeError", "raw_content": final_response_content, "total_score": 0, "star_rating": 0}
                if output_path:
                    append_single_result_to_csv(participant_data, result, output_path)
                return result
        else: 
            print(f"Error: No final content from Orchestrator for {participant_data.get('name', 'N/A')} (user_id: {user_id})")
            result = {"error": "No final content from Orchestrator", "total_score": 0, "star_rating": 0}
            if output_path:
                append_single_result_to_csv(participant_data, result, output_path)
            return result
    except json.JSONDecodeError as e: 
        print(f"Error: Could not decode JSON from Orchestrator for {participant_data.get('name', 'N/A')}. Content: '{final_response_content}'. Error: {e}")
        result = {"error": "JSONDecodeError", "raw_content": final_response_content, "total_score": 0, "star_rating": 0}
        if output_path:
            append_single_result_to_csv(participant_data, result, output_path)
        return result
    except Exception as e: 
        print(f"Error processing participant {participant_data.get('name', 'N/A')} with Orchestrator: {e}")
        import traceback
        traceback.print_exc()
        result = {"error": str(e), "total_score": 0, "star_rating": 0}
        if output_path:
            append_single_result_to_csv(participant_data, result, output_path)
        return result

async def process_batch(
    batch_df: pd.DataFrame,
    runner: Runner,
    session_service: InMemorySessionService,
    app_name: str,
    semaphore: asyncio.Semaphore,
    batch_num: int,
    total_participants: int,
    output_path: str
) -> List[Dict]:
    batch_results = []
    tasks = []
    
    for index, row in batch_df.iterrows():
        async def process_with_semaphore(row=row, index=index):
            async with semaphore:
                # Each participant is now written to CSV immediately after processing
                return await process_single_participant_multi_agent(row, runner, session_service, app_name, index, output_path)
        
        tasks.append(process_with_semaphore())
    
    batch_start = (batch_num - 1) * len(tasks)
    batch_end = batch_start + len(tasks)
    print(f"Processing batch {batch_num} with {len(tasks)} participants ({batch_start + 1}-{batch_end} of {total_participants})...")
    results = await asyncio.gather(*tasks)
    
    # Results are already written to CSV, just collect them for summary
    for row, result in zip(batch_df.iterrows(), results):
        merged_result = row[1].to_dict()
        if isinstance(result, dict):
            # Only include non-error results and exclude error/raw_content fields
            if "error" not in result:
                # Filter out error-related fields
                result = {k: v for k, v in result.items() if k not in ['error', 'raw_content', 'github_fetch_error']}
                merged_result.update(result)
        batch_results.append(merged_result)
    
    processed_so_far = batch_num * BATCH_SIZE
    if processed_so_far > total_participants:
        processed_so_far = total_participants
    print(f"✓ Batch {batch_num} complete. Progress: {processed_so_far}/{total_participants} participants processed ({(processed_so_far/total_participants)*100:.1f}%)")
    
    return batch_results

async def process_all_batches(
    input_df: pd.DataFrame,
    runner: Runner,
    session_service: InMemorySessionService,
    app_name: str
) -> List[Dict]:
    all_results = []
    total_participants = len(input_df)
    num_batches = math.ceil(total_participants / BATCH_SIZE)
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
    
    PARALLEL_BATCHES = 5  # Process 5 batches at once
    
    # Process batches in groups of PARALLEL_BATCHES
    for group_start in range(0, num_batches, PARALLEL_BATCHES):
        group_end = min(group_start + PARALLEL_BATCHES, num_batches)
        batch_tasks = []
        
        for batch_num in range(group_start, group_end):
            start_idx = batch_num * BATCH_SIZE
            end_idx = min((batch_num + 1) * BATCH_SIZE, total_participants)
            batch_df = input_df.iloc[start_idx:end_idx].copy()
            
            print(f"Queuing batch {batch_num + 1}/{num_batches} ({start_idx + 1}-{end_idx} of {total_participants})")
            batch_tasks.append(process_batch(
                batch_df, runner, session_service, app_name, semaphore, batch_num + 1, total_participants, OUTPUT_CSV_PATH
            ))
        
        print(f"\nProcessing {len(batch_tasks)} batches in parallel...")
        batch_results_list = await asyncio.gather(*batch_tasks)
        
        for batch_results in batch_results_list:
            all_results.extend(batch_results)
        
        participants_processed = len(all_results)
        print(f"✓ Completed batch group {group_start + 1}-{group_end} of {num_batches}. Total progress: {participants_processed}/{total_participants} participants ({(participants_processed/total_participants)*100:.1f}%)")
    
    return all_results

def get_processed_emails(output_path: str) -> set:
    """
    Read the output CSV and return a set of emails that have already been processed.
    """
    if not os.path.exists(output_path):
        return set()
    
    try:
        processed_df = pd.read_csv(output_path)
        if 'Email' in processed_df.columns:
            # Return set of lowercase emails for case-insensitive comparison
            return set(processed_df['Email'].astype(str).str.lower())
        return set()
    except Exception as e:
        print(f"Warning: Could not read existing output file for resume: {e}")
        return set()

async def test_single_participant_by_email(
    email_to_test: str,
    df: pd.DataFrame,
    runner: Runner, 
    session_service: InMemorySessionService, 
    app_name: str
):
    # (No change in this function's internal logic)
    participant_row_df = df[df['Email'].str.lower() == email_to_test.lower()] # participant_row is a DataFrame
    if participant_row_df.empty: print(f"\n--- Test Mode ---\nParticipant with email '{email_to_test}' not found.\n--- End Test ---\n"); return
    participant_data = participant_row_df.iloc[0] 
    # Find the original index of this participant in the main DataFrame to use for unique user_id/session_id
    try:
        original_index = participant_row_df.index[0]
    except IndexError: # Should not happen if participant_row_df is not empty
        original_index = -1 # Fallback, less ideal for session uniqueness
        
    print(f"\n--- Testing Participant: {participant_data.get('Name', 'N/A')} ({email_to_test}) ---")
    print("\nParticipant Input Data:"); [print(f"  {key}: {value}") for key, value in participant_data.items()]
    print("\nScoring participant via Runner...")
    scoring_result = await process_single_participant_multi_agent(participant_data, runner, session_service, app_name, original_index, None)
    print("\nScoring Result (JSON from Orchestrator):"); print(json.dumps(scoring_result, indent=2))
    if isinstance(scoring_result, dict) and "error" not in scoring_result:
        print("\nScore Summary:"); print(f"  Total Score: {scoring_result.get('total_score', 'N/A')}"); print(f"  Star Rating: {scoring_result.get('star_rating', 'N/A')}")
        print(f"  GitHub Score: {scoring_result.get('github_score', 'N/A')} (Contributions: {scoring_result.get('github_contributions', 'N/A')}, User: {scoring_result.get('github_username', 'N/A')})")
        if scoring_result.get('github_fetch_error'): print(f"  GitHub Error: {scoring_result.get('github_fetch_error')}")
    elif isinstance(scoring_result, dict) and "error" in scoring_result: 
        print(f"  Error during scoring: {scoring_result['error']}")
        if "raw_content" in scoring_result: 
            print(f"  Raw LLM Content (on error): {scoring_result['raw_content']}")
    print(f"--- End Test for {email_to_test} ---\n")

async def main():
    if not GOOGLE_API_KEY: print("Warning: GOOGLE_API_KEY environment variable not set.")
    if not GITHUB_PAT: print("Warning: GITHUB_PAT environment variable not set.")

    # Only LLM agents for questions, status is now a tool function
    q1_agent = LlmAgent(name="Q1ScoringAgent", model="gemini-2.5-flash", instruction=PROMPT_Q1, description="Scores Q1 (AI Hallucination).")
    q2_agent = LlmAgent(name="Q2ScoringAgent", model="gemini-2.5-flash", instruction=PROMPT_Q2, description="Scores Q2 (Multi-agent Hallucination Prevention).")
    q3_agent = LlmAgent(name="Q3ScoringAgent", model="gemini-2.5-flash", instruction=PROMPT_Q3, description="Scores Q3 (Function Calling).")
    q4_agent = LlmAgent(name="Q4ScoringAgent", model="gemini-2.5-flash", instruction=PROMPT_Q4, description="Scores Q4 (RAG vs Fine-tuning).")
    q5_agent = LlmAgent(name="Q5ScoringAgent", model="gemini-2.5-flash", instruction=PROMPT_Q5_EXPECTATIONS, description="Scores Q5 (Expectations).")

    # Create tool instances for both rule-based functions
    status_tool_instance = FunctionTool(func=status_type_scoring_tool_func)
    github_tool_instance = FunctionTool(func=github_processing_tool_func)

    tools_for_orchestrator = [
        status_tool_instance,  # Status is now a tool function
        agent_tool.AgentTool(agent=q1_agent),
        agent_tool.AgentTool(agent=q2_agent),
        agent_tool.AgentTool(agent=q3_agent),
        agent_tool.AgentTool(agent=q4_agent),
        agent_tool.AgentTool(agent=q5_agent),
        github_tool_instance,
    ]

    orchestrator_agent = LlmAgent(
        name="ParticipantScoreOrchestratorFinal", model="gemini-2.5-pro",
        instruction="You are a master evaluator. Use the provided tools to call specialist agents/functions for each scoring criterion by their names (e.g., 'status_type_scoring_tool_func' for status scoring, 'Q1ScoringAgent' for Q1 agent, 'github_processing_tool_func' for the github function tool). Provide the exact arguments as specified for each tool in the detailed user prompt. Then, aggregate the results as specified.",
        tools=tools_for_orchestrator
    )
    
    session_service = InMemorySessionService()
    runner = Runner(agent=orchestrator_agent, app_name=APP_NAME, session_service=session_service)

    try: input_df = pd.read_csv(INPUT_CSV_PATH); print(f"Successfully loaded {INPUT_CSV_PATH}, {len(input_df)} records found.")
    except Exception as e: print(f"Error reading CSV: {e}"); return
    
    expected_cols = ['Email', 'Name', 'Status Type', 'GitHub', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    for col_idx, col in enumerate(expected_cols): # Use enumerate for better placeholder naming if needed
        if col not in input_df.columns:
            placeholder_name = f"placeholder_col_{col_idx}" # Use a generic placeholder name
            print(f"Warning: Column '{col}' (expected as '{placeholder_name}' if original name missing) not found in CSV. Adding it as empty.")
            input_df[col] = "" # Add column with original expected name to avoid downstream errors
        input_df[col] = input_df[col].astype(str).fillna("")
    if 'Email' not in input_df.columns:
        print("Critical Error: 'Email' column is absolutely required and missing from the CSV. Cannot proceed with testing or processing.")
        return

    run_full_processing = True

    if run_full_processing:
        # Check for existing processed participants to enable resume
        processed_emails = get_processed_emails(OUTPUT_CSV_PATH)
        
        if processed_emails:
            print(f"\n{'='*60}")
            print(f"RESUME MODE: Found {len(processed_emails)} already processed participants")
            print(f"{'='*60}")
            # Filter out already processed participants (case-insensitive)
            input_df['Email_lower'] = input_df['Email'].astype(str).str.lower()
            unprocessed_df = input_df[~input_df['Email_lower'].isin(processed_emails)].copy()
            unprocessed_df = unprocessed_df.drop(columns=['Email_lower'])
            
            skipped_count = len(input_df) - len(unprocessed_df)
            print(f"Skipping {skipped_count} already processed participants")
            print(f"Resuming with {len(unprocessed_df)} remaining participants")
            print(f"{'='*60}\n")
            
            if len(unprocessed_df) == 0:
                print("All participants have already been processed!")
                return
            
            input_df = unprocessed_df
        else:
            print(f"\nStarting fresh processing of {len(input_df)} participants...")
            
        try:
            print(f"\nProcessing {len(input_df)} participants...")
            results_list = await process_all_batches(input_df, runner, session_service, APP_NAME)
            print(f"\n{'='*60}")
            print(f"✓ COMPLETED: Successfully processed all {len(results_list)} participants")
            print(f"{'='*60}")
        except KeyboardInterrupt:
            processed_count = len(get_processed_emails(OUTPUT_CSV_PATH))
            print(f"\n\n{'='*60}")
            print(f"⚠ INTERRUPTED: Processing stopped by user")
            print(f"{'='*60}")
            print(f"Processed: {processed_count} participants")
            print(f"Saved to: {OUTPUT_CSV_PATH}")
            print(f"\nTo resume, simply run the script again.")
            print(f"The system will automatically skip already processed participants.")
            print(f"{'='*60}")
            raise
        except Exception as e:
            processed_count = len(get_processed_emails(OUTPUT_CSV_PATH))
            print(f"\n\n{'='*60}")
            print(f"⚠ ERROR: Processing stopped due to error")
            print(f"{'='*60}")
            print(f"Error: {e}")
            print(f"Processed: {processed_count} participants before error")
            print(f"Saved to: {OUTPUT_CSV_PATH}")
            print(f"\nTo resume, simply run the script again.")
            print(f"The system will automatically skip already processed participants.")
            print(f"{'='*60}")
            raise
    else:
        print("Full CSV processing was skipped.")

if __name__ == "__main__":
    asyncio.run(main())