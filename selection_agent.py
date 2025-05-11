import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import os
import re
import asyncio

# ADK Imports - Adjusted based on user example
from google.adk.tools import FunctionTool, agent_tool
from google.adk.agents import LlmAgent 
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from pydantic import BaseModel, Field

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GITHUB_PAT = os.environ.get("GITHUB_PAT")

INPUT_CSV_PATH = "bwai_reg_updated.csv"
OUTPUT_CSV_PATH = "bwai_reg_multi_agent_runner_scored_with_test.csv" # New name
APP_NAME = "ParticipantScoringApp"

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

# --- Rule-Based Scoring Functions --- (No change)
def _score_github_contributions_rule_based(contributions: int) -> int:
    if contributions > 1000: return 50
    if contributions > 500: return 20
    if contributions > 200: return 15
    if contributions > 50:  return 10
    if contributions >= 10: return 5
    return 0
def _score_sq5_vertex_rule_based(response: str) -> int:
    if not isinstance(response, str): response = ""
    response = response.lower().strip();
    if "yes, very interested" in response or "yes, specific interest" in response or "very interested" in response : return 10
    if "yes" == response or ("yes" in response and len(response) < 20) : return 6
    if "maybe" in response or "neutral" in response or "somewhat interested" in response : return 2
    return 0
def _score_attendance_rule_based(attendance_text: str) -> int:
    if not isinstance(attendance_text, str): attendance_text = ""
    attendance_text = attendance_text.lower().strip();
    if "both days" in attendance_text or "both" == attendance_text : return 5
    if "day1" in attendance_text or "day2" in attendance_text or "one day" in attendance_text: return 2
    return 0

# --- Tool Function Implementations --- (No change)
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

def sq5_vertex_scoring_tool_func(sq5_response: str) -> dict:
    """Scores SQ5 (Vertex AI interest) based on rules. Output keys: "score"."""
    return {"score": _score_sq5_vertex_rule_based(sq5_response)}

def attendance_scoring_tool_func(attendance_text: str) -> dict:
    """Scores attendance preference based on rules. Output keys: "score"."""
    return {"score": _score_attendance_rule_based(attendance_text)}

# --- Specialist Agent Input/Output Schemas (Pydantic models for agent_tool wrappers) --- (No change)
class SingleTextEvalInputArgs(BaseModel):
    text_to_evaluate: str
class SQ1ExpEvalInputArgs(BaseModel):
    sq1_response: str
    expectations: str
class ScoreOutput(BaseModel):
    score: int = Field(description="The calculated integer score for the criterion.")

# --- Specialist LLM Agent Prompts (Content no change) ---
PROMPT_OCCUPATION = """You are an AI assistant evaluating a participant's occupation.
The occupation text is provided in the 'text_to_evaluate' argument.
Score based on these tiers:
- Tier 1 (30 pts): Highly Relevant Professional Roles (e.g., AI/ML Engineer, Data Scientist, ...).
- Tier 2 (20 pts): Relevant Professional Roles (e.g., Software Engineer, ...).
- Tier 3 (10 pts): Aspiring Professionals/Academic (e.g., Recent Graduate, ...).
- Tier 4 (5 pts): Career Changers/Enthusiasts/Other Professionals.
- Tier 5 (0 pts): Less Relevant/Unclear or no occupation provided.
Return ONLY a JSON object: {{"score": <integer_score>}}. If occupation is unscorable, return {{"score": 0}}.
"""
PROMPT_SQ1_EXPECTATIONS = """You are an AI evaluating project excitement and expectations for an AI conference.
The SQ1 response is in 'sq1_response' argument and expectations in 'expectations' argument.
Conference sessions cover: Firebase AI, Genkit, Connected Agents, ADK, GraphRAG, Gemma, Neural Network control, AI for Ops, Gemini TTS, Kubernetes LLMs, Secure AI.
Score based on BOTH texts for alignment and enthusiasm for hands-on technical learning:
- 10-15 pts: Strong alignment with specific conference session topics. Desire to build/apply.
- 5-9 pts: General AI/ML interest. Upskilling goals. Proactive but not tied to specific sessions.
- 1-4 pts: Vague interest. Generic learning/networking. Not closely related to technical depth.
- 0 pts: No relevant information.
Return ONLY a JSON object: {{"score": <integer_score>}}.
"""
PROMPT_SDK_SQ2 = """Evaluate Gen AI SDK (Python) usage from the response provided in 'text_to_evaluate' argument.
- Extensive use/projects: 10 points.
- Simple "Yes"/familiar/tried: 6 points.
- "No, but eager/aware": 2 points.
- "No"/no answer: 0 points.
Return ONLY a JSON object: {{"score": <integer_score>}}.
"""
PROMPT_COLAB_SQ3 = """Evaluate Google Colab comfort from the response in 'text_to_evaluate' argument.
- "Very comfortable": 5 points.
- "Comfortable" (not "Somewhat"/"Not"): 3 points.
- "Somewhat comfortable": 1 point.
- "Not comfortable"/no answer: 0 points.
Return ONLY a JSON object: {{"score": <integer_score>}}.
"""
PROMPT_INDUSTRY_SQ4 = """Evaluate primary generative AI application domain from the response in 'text_to_evaluate' argument.
Relevant: software dev, healthcare AI, finance, automation, enterprise, edutech, creative.
- Clearly defined & relevant industry: 3-5 points.
- Broad interest/multiple areas: 2 points.
- Vague/personal projects: 1 point.
- No answer/irrelevant: 0 points.
Return ONLY a JSON object: {{"score": <integer_score>}}.
"""

# --- Orchestrator Agent Prompt Template (No change in content) ---
ORCHESTRATOR_PROMPT_TEMPLATE = """
You are a master evaluator for AI conference participants. For the given participant data, your goal is to determine all individual scores by calling specialized scoring agents/tools, then calculate the total score and star rating.

**Participant Data:**
Name: {name}
Email: {email}
Occupation: {occupation}
Attendance Dates: {attendance_dates}
GitHub Profile URL: {github_profile_url}
SQ1 Response (Hands-on projects): {sq1_response}
Expectations: {expectations}
SQ2 Response (Google Gen AI SDK): {sq2_response}
SQ3 Response (Google Colab comfort): {sq3_response}
SQ4 Response (Industry/Domain): {sq4_response}
SQ5 Response (Vertex AI interest): {sq5_response}

**Evaluation Steps & Tool Usage:**
1.  Call the `OccupationScoringAgent` tool. Provide the argument `text_to_evaluate` set to the participant's `occupation` text. Expect JSON `{{ "score": <num> }}`. Let this be `occupation_score_json`.
2.  Call the `attendance_scoring_tool_func` tool. Provide the argument `attendance_text` set to the participant's `attendance_dates` text. Expect JSON `{{ "score": <num> }}`. Let this be `attendance_score_json`.
3.  Call the `github_processing_tool_func` tool. Provide the argument `github_profile_url` set to participant's `github_profile_url`. Expect JSON `{{ "github_score": <num>, "github_username": <str_or_null>, "github_contributions": <num>, "github_fetch_error": <str_or_null> }}`. Let this be `github_details_json`.
4.  Call the `SQ1ExpectationsScoringAgent` tool. Provide arguments `sq1_response` (set to participant's SQ1 text) and `expectations` (set to participant's expectations text). Expect JSON `{{ "score": <num> }}`. Let this be `sq1_exp_score_json`.
5.  Call the `SDKUsageScoringAgent` tool. Provide argument `text_to_evaluate` set to `sq2_response`. Expect JSON `{{ "score": <num> }}`. Let this be `sdk_score_json`.
6.  Call the `ColabComfortScoringAgent` tool. Provide argument `text_to_evaluate` set to `sq3_response`. Expect JSON `{{ "score": <num> }}`. Let this be `colab_score_json`.
7.  Call the `IndustryDomainScoringAgent` tool. Provide argument `text_to_evaluate` set to `sq4_response`. Expect JSON `{{ "score": <num> }}`. Let this be `sq4_score_json`.
8.  Call the `sq5_vertex_scoring_tool_func` tool. Provide argument `sq5_response` set to participant's `sq5_response`. Expect JSON `{{ "score": <num> }}`. Let this be `vertex_score_json`.

**Data Aggregation and Final Calculation:**
* `occupation_score` = `occupation_score_json['score']`
* `attendance_score` = `attendance_score_json['score']`
* `github_score` = `github_details_json['github_score']` (also get `github_username`, `github_contributions`, `github_fetch_error` from `github_details_json`)
* `sq1_exp_score` = `sq1_exp_score_json['score']`
* `sdk_score` = `sdk_score_json['score']`
* `colab_score` = `colab_score_json['score']`
* `sq4_score` = `sq4_score_json['score']`
* `vertex_score` = `vertex_score_json['score']`
* Sum all these scores to get `total_score`.
* Calculate an integer `star_rating` (0-5) based on `total_score`: 90-100=5, 75-89=4, 60-74=3, 40-59=2, 1-39=1, 0=0.

**Output Format:**
Return a single JSON object with ALL the following keys:
"github_username", "github_contributions", "github_score", "occupation_score", "attendance_score", "sdk_score", "colab_score", "vertex_score", "sq1_exp_score", "sq4_score", "total_score", "star_rating", "github_fetch_error".
Ensure all score values and star_rating are integers.
"""

async def process_single_participant_multi_agent(
    participant_data: pd.Series,
    runner: Runner, 
    session_service: InMemorySessionService, 
    app_name: str,
    participant_index: int 
) -> dict:
    # (No change in this function's internal logic)
    prompt = ORCHESTRATOR_PROMPT_TEMPLATE.format(
        name=participant_data.get('name', 'N/A'), email=participant_data.get('email', 'N/A'),
        occupation=str(participant_data.get('occupation', '')), attendance_dates=str(participant_data.get('attendance_dates', '')),
        github_profile_url=str(participant_data.get('github_profile', '')), sq1_response=str(participant_data.get('screening_responses/0/response', '')),
        expectations=str(participant_data.get('expectations', '')), sq2_response=str(participant_data.get('screening_responses/1/response', '')),
        sq3_response=str(participant_data.get('screening_responses/2/response', '')), sq4_response=str(participant_data.get('screening_responses/3/response', '')),
        sq5_response=str(participant_data.get('screening_responses/4/response', ''))
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
            if final_response_content.startswith("```json"): final_response_content = final_response_content[7:]
            if final_response_content.endswith("```"): final_response_content = final_response_content[:-3]
            response_json = json.loads(final_response_content.strip())
            if 'star_rating' in response_json and isinstance(response_json['star_rating'], str):
                try: response_json['star_rating'] = int(response_json['star_rating'])
                except ValueError: response_json['star_rating'] = 0
            return response_json
        else: print(f"Error: No final content from Orchestrator for {participant_data.get('name', 'N/A')} (user_id: {user_id})"); return {"error": "No final content from Orchestrator", "total_score": 0, "star_rating": 0}
    except json.JSONDecodeError as e: print(f"Error: Could not decode JSON from Orchestrator for {participant_data.get('name', 'N/A')}. Content: '{final_response_content}'. Error: {e}"); return {"error": "JSONDecodeError", "raw_content": final_response_content, "total_score": 0, "star_rating": 0}
    except Exception as e: print(f"Error processing participant {participant_data.get('name', 'N/A')} with Orchestrator: {e}"); import traceback; traceback.print_exc(); return {"error": str(e), "total_score": 0, "star_rating": 0}

async def test_single_participant_by_email(
    email_to_test: str,
    df: pd.DataFrame,
    runner: Runner, 
    session_service: InMemorySessionService, 
    app_name: str
):
    # (No change in this function's internal logic)
    participant_row_df = df[df['email'].str.lower() == email_to_test.lower()] # participant_row is a DataFrame
    if participant_row_df.empty: print(f"\n--- Test Mode ---\nParticipant with email '{email_to_test}' not found.\n--- End Test ---\n"); return
    participant_data = participant_row_df.iloc[0] 
    # Find the original index of this participant in the main DataFrame to use for unique user_id/session_id
    try:
        original_index = participant_row_df.index[0]
    except IndexError: # Should not happen if participant_row_df is not empty
        original_index = -1 # Fallback, less ideal for session uniqueness
        
    print(f"\n--- Testing Participant: {participant_data.get('name', 'N/A')} ({email_to_test}) ---")
    print("\nParticipant Input Data:"); [print(f"  {key}: {value}") for key, value in participant_data.items()]
    print("\nScoring participant via Runner...")
    scoring_result = await process_single_participant_multi_agent(participant_data, runner, session_service, app_name, original_index)
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

    occupation_agent = LlmAgent(name="OccupationScoringAgent", model="gemini-2.5-flash-preview-04-17", instruction=PROMPT_OCCUPATION, description="Scores participant's occupation.")
    sq1_exp_agent = LlmAgent(name="SQ1ExpectationsScoringAgent", model="gemini-2.5-flash-preview-04-17", instruction=PROMPT_SQ1_EXPECTATIONS, description="Scores SQ1 and Expectations.")
    sdk_sq2_agent = LlmAgent(name="SDKUsageScoringAgent", model="gemini-2.5-flash-preview-04-17", instruction=PROMPT_SDK_SQ2, description="Scores SQ2 (SDK usage).")
    colab_sq3_agent = LlmAgent(name="ColabComfortScoringAgent", model="gemini-2.5-flash-preview-04-17", instruction=PROMPT_COLAB_SQ3, description="Scores SQ3 (Colab comfort).")
    industry_sq4_agent = LlmAgent(name="IndustryDomainScoringAgent", model="gemini-2.5-flash-preview-04-17", instruction=PROMPT_INDUSTRY_SQ4, description="Scores SQ4 (Industry/Domain).")

    github_tool_instance = FunctionTool(func=github_processing_tool_func)
    sq5_tool_instance = FunctionTool(func=sq5_vertex_scoring_tool_func)
    attendance_tool_instance = FunctionTool(func=attendance_scoring_tool_func)

    tools_for_orchestrator = [
        agent_tool.AgentTool(agent=occupation_agent), agent_tool.AgentTool(agent=sq1_exp_agent),
        agent_tool.AgentTool(agent=sdk_sq2_agent), agent_tool.AgentTool(agent=colab_sq3_agent),
        agent_tool.AgentTool(agent=industry_sq4_agent),
        github_tool_instance, sq5_tool_instance, attendance_tool_instance,
    ]

    orchestrator_agent = LlmAgent(
        name="ParticipantScoreOrchestratorFinal", model="gemini-1.5-pro-latest",
        instruction="You are a master evaluator. Use the provided tools to call specialist agents/functions for each scoring criterion by their names (e.g., 'OccupationScoringAgent' for the agent, 'github_processing_tool_func' for the function tool). Provide the exact arguments as specified for each tool in the detailed user prompt. Then, aggregate the results as specified.",
        tools=tools_for_orchestrator
    )
    
    session_service = InMemorySessionService()
    runner = Runner(agent=orchestrator_agent, app_name=APP_NAME, session_service=session_service)

    try: input_df = pd.read_csv(INPUT_CSV_PATH); print(f"Successfully loaded {INPUT_CSV_PATH}, {len(input_df)} records found.")
    except Exception as e: print(f"Error reading CSV: {e}"); return
    
    expected_cols = ['email', 'name', 'occupation', 'attendance_dates', 'github_profile', 'expectations', 'screening_responses/0/response', 'screening_responses/1/response', 'screening_responses/2/response', 'screening_responses/3/response', 'screening_responses/4/response']
    for col_idx, col in enumerate(expected_cols): # Use enumerate for better placeholder naming if needed
        if col not in input_df.columns:
            placeholder_name = f"placeholder_col_{col_idx}" # Use a generic placeholder name
            print(f"Warning: Column '{col}' (expected as '{placeholder_name}' if original name missing) not found in CSV. Adding it as empty.")
            input_df[col] = "" # Add column with original expected name to avoid downstream errors
        input_df[col] = input_df[col].astype(str).fillna("")
    if 'email' not in input_df.columns:
        print("Critical Error: 'email' column is absolutely required and missing from the CSV. Cannot proceed with testing or processing.")
        return


    # --- << TEST SECTION >> ---
    # Replace with a real email from your CSV to activate the test.
    # Keep the placeholder to prevent accidental runs on it.
    email_to_test = "geesadbandara25@gmail.com"  # <<== REPLACE THIS WITH AN ACTUAL EMAIL FROM YOUR CSV

    run_full_processing = False # Flag to control full processing after test

    if 'email' in input_df.columns:
        if email_to_test != "participant_email_to_test@example.com": # Check if user changed the placeholder
            print(f"--- Attempting Test for: {email_to_test} ---")
            await test_single_participant_by_email(email_to_test, input_df, runner, session_service, APP_NAME)
            # Ask user if they want to continue with full processing or exit
            # This is a simple CLI interaction, for a real app this would be different
            # proceed = input("Test finished. Continue with full CSV processing? (yes/no): ").lower()
            # if proceed != 'yes':
            #     run_full_processing = False
            #     print("Exiting after single test.")
            # For non-interactive script, decide behavior:
            run_full_processing = False # Default to False to avoid accidental full run after a specific test
            print("Single participant test complete. To run full processing, comment out the test call or set run_full_processing = True.")

        else:
            print("INFO: Test email is still the placeholder. Skipping single participant test. To test, update 'email_to_test'.")
    else:
         print("WARNING: Test function skipped: 'email' column missing in CSV.")
    # --- << END TEST SECTION >> ---

    if run_full_processing:
        results_list = []
        print(f"\nStarting full multi-agent scoring (runner-based) of {len(input_df)} participants...")
        for index, row in input_df.iterrows():
            print(f"Processing participant {index + 1}/{len(input_df)}: {row.get('name', 'N/A')} ({row.get('email', 'N/A')})")
            participant_result = await process_single_participant_multi_agent(row, runner, session_service, APP_NAME, index)
            merged_result = row.to_dict(); merged_result.update(participant_result if isinstance(participant_result, dict) else {})
            default_scores = { "github_username": None, "github_contributions": 0, "github_score": 0, "occupation_score": 0, "attendance_score": 0, "sdk_score": 0, "colab_score": 0, "vertex_score": 0, "sq1_exp_score": 0, "sq4_score": 0, "total_score": 0, "star_rating": 0, "github_fetch_error": None }
            for key, default_value in default_scores.items():
                if key not in merged_result: merged_result[key] = default_value
                if key == 'star_rating' and isinstance(merged_result[key], str):
                    try: merged_result[key] = int(merged_result[key])
                    except ValueError: merged_result[key] = 0
            results_list.append(merged_result)

        print(f"\nFinished scoring {len(results_list)} participants.")
        output_df = pd.DataFrame(results_list)
        score_cols_to_numeric = [ "github_contributions", "github_score", "occupation_score", "attendance_score", "sdk_score", "colab_score", "vertex_score", "sq1_exp_score", "sq4_score", "total_score", "star_rating" ]
        for col in score_cols_to_numeric:
            if col in output_df.columns: output_df[col] = pd.to_numeric(output_df[col], errors='coerce').fillna(0).astype(int)
            else: output_df[col] = 0
        if 'total_score' in output_df.columns: output_df.sort_values(by='total_score', ascending=False, inplace=True)
        try: output_df.to_csv(OUTPUT_CSV_PATH, index=False); print(f"Processing complete. Output saved to {OUTPUT_CSV_PATH}")
        except Exception as e: print(f"Error writing output CSV: {e}")
    else:
        print("Full CSV processing was skipped.")

if __name__ == "__main__":
    asyncio.run(main())