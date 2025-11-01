import streamlit as st
import sys
import os
import sqlalchemy
import pandas as pd
import time
import datetime # <--- FIX: Explicitly importing datetime
from contextlib import redirect_stdout
import io
import traceback # For detailed error printing
from typing import Dict, Any

# --- Add project root to path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(PROJECT_ROOT) == 'src':
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# --------------------------------
try:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
    from src import config, tools, agent
except ImportError as e:
    st.error(f"Error importing modules: {e}. Make sure 'src' directory exists and is in the correct path relative to app.py.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during import: {e}")
    st.stop()

# --- Page Config ---
st.set_page_config(
    page_title="Sarathī-Chariot: Road Safety Expert Agent",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_agent():
    try:
        chain = agent.create_supervisor_chain()
        if hasattr(chain, 'llm'):
            st.sidebar.success(f"LLM initialized: {config.OLLAMA_MODEL}")
        return chain
    except Exception as e:
        st.error(f"Failed to load the agent: {e}. Check agent.py and dependencies.")
        st.stop()
        return None

@st.cache_data
def get_filter_options():
    try:
        if not os.path.exists(config.STRUCTURED_DB_PATH):
            st.warning(f"Database not found at {config.STRUCTURED_DB_PATH}. Please run src/data_processor.py first.")
            return [], [], []
        engine = sqlalchemy.create_engine(f"sqlite:///{config.STRUCTURED_DB_PATH}")
        with engine.connect() as conn:
            problems = [row[0] for row in conn.execute(sqlalchemy.text(f"SELECT DISTINCT problem FROM {config.SQL_TABLE_NAME} ORDER BY problem"))]
            categories = [row[0] for row in conn.execute(sqlalchemy.text(f"SELECT DISTINCT category FROM {config.SQL_TABLE_NAME} ORDER BY category"))]
            types = [row[0] for row in conn.execute(sqlalchemy.text(f"SELECT DISTINCT type FROM {config.SQL_TABLE_NAME} ORDER BY type"))]
        if not problems:
            st.warning("Could not load filter options. Database might be empty or inaccessible.")
            return [], [], []
        return problems, categories, types
    except Exception as e:
        st.error(f"Error loading filter options from database: {e}")
        return [], [], []

def load_knowledge_gaps():
    try:
        if not os.path.exists(config.KNOWLEDGE_GAP_LOG_PATH):
            return "No knowledge gaps logged yet. Queries that yield no results will be logged here."
        with open(config.KNOWLEDGE_GAP_LOG_PATH, "r") as f:
            content = f.read()
            return content if content else "Log file is empty."
    except Exception as e:
        return f"Error loading log: {e}"

def log_feedback(query: str, response: str, rating: str, exec_time: float):
    """Logs user feedback to a dedicated file."""
    feedback_path = os.path.join(PROJECT_ROOT, 'data', 'knowledge_gaps.log')
    try:
        os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
        with open(feedback_path, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"--- FEEDBACK --- [{timestamp} | Time: {exec_time:.2f}s | Rating: {rating}]\n")
            f.write(f"Query: {query}\n")
            f.write(f"Response Snippet: {response[:150]}...\n\n")
        st.toast(f"Feedback logged: {rating}!", icon="👍")
    except Exception as e:
        st.warning(f"Could not log feedback: {e}")


# --- Load Data & Agent ---
agent_chain = load_agent()
problems, categories, types = get_filter_options()


# --- Custom CSS for Aesthetics (Minimal changes) ---
st.markdown("""
<style>
/* Chat input appearance */
.stChatInput > div > div {
    border-radius: 0.5rem;
    border: 1px solid #e0e0e0;
}
/* Chat window styling to make sure it looks nice */
[data-testid="stChatMessage"] {
    background-color: #f0f2f6; /* Light grey background for messages */
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="stChatMessage"] p { 
    margin: 0; /* Remove default paragraph margins */
}
</style>
""", unsafe_allow_html=True)

# --- [NEW] Standard Title and Caption (Replaces sticky header) ---
st.title("Sarathī 🛞: Road Safety Expert Agent")
st.caption(f"Query the Indian Road Congress (IRC) standards with AI. Powered by Local RAG & **{config.OLLAMA_MODEL}**.")


# --- Sidebar: Engineer (Filter) Mode ---
with st.sidebar:
    st.markdown("<h2>🔍 Engineer Mode (Filters)</h2>", unsafe_allow_html=True)
    st.markdown("Use precise filters to query the structured IRC standards database.")

    with st.container(border=True):
        if not problems and not categories and not types:
            st.error("Filter options failed to load. Cannot use Engineer Mode.")
        else:
            selected_problem = st.selectbox("🛑 Problem Type", options=["Any"] + problems, key="filter_problem")
            selected_category = st.selectbox("🛣️ Asset Category", options=["Any"] + categories, key="filter_category")
            selected_type = st.selectbox("🚧 Sign/Marking Type", options=["Any"] + types, key="filter_type")

            if st.button("Search Database", use_container_width=True, key="filter_search"):
                filters = {}
                if selected_problem != "Any": filters["problem"] = selected_problem
                if selected_category != "Any": filters["category"] = selected_category
                if selected_type != "Any": filters["type"] = selected_type

                if not filters:
                    st.warning("Please select at least one filter.")
                else:
                    st.session_state.filter_results = {"filters": filters, "results": None, "error": None}
                    with st.spinner("Querying structured database..."):
                        try:
                            st.session_state.filter_results["results"] = tools.structured_search(**filters)
                        except Exception as e:
                            st.session_state.filter_results["error"] = f"Error during structured search: {e}"
                            
                    # [NEW] Rerun to update the expander in tab1
                    st.rerun()

    st.markdown("---")
    st.info(f"Powered by Local RAG & **{config.OLLAMA_MODEL}** via Ollama.")


# --- Main Content Tabs ---
tab1, tab2 = st.tabs(["💬 Expert Chat", "📄 Knowledge Gap Log"])


with tab1:
    # --- Chat History Container (PRIMARY FOCUS) ---
    # [MODIFIED] Set height to a value that works for most screens, e.g., 500px
    chat_container = st.container(height=500, border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.exec_time = None

    # Display chat messages from history inside the container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # --- Input Handling Logic (STAYS IMMEDIATELY BELOW CHAT CONTAINER) ---
    user_prompt = st.chat_input("What is your road safety question?", key="chat_input")
    prompt_to_process = None

    # --- Auxiliary Metrics and Feedback (Consolidated Section) ---
    st.markdown("---")
    
    col_time, col_feedback = st.columns([0.3, 0.7])
    
    with col_time:
        if st.session_state.exec_time:
            st.info(f"⏱️ **Agent Time:** {st.session_state.exec_time:.2f} seconds")
        else:
            st.info("⏱️ **Agent Time:** N/A")
        
        # [NEW] Moved checkbox here for a cleaner layout
        show_thoughts = st.checkbox("Show Agent Thoughts", value=False, key="show_thoughts")

    with col_feedback:
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            with st.form("feedback_form", clear_on_submit=True, border=True):
                st.markdown("###### Did this answer satisfy your query?")
                feedback_cols = st.columns(3)
                
                satisfied = feedback_cols[0].form_submit_button("Satisfied 😊", use_container_width=True)
                average = feedback_cols[1].form_submit_button("Average 😐", use_container_width=True)
                not_satisfied = feedback_cols[2].form_submit_button("Not Satisfied 😔", use_container_width=True)
                
                last_query = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else "N/A"
                last_response = st.session_state.messages[-1]["content"] if st.session_state.messages else "N/A"
                
                if satisfied or average or not_satisfied:
                    rating = "Satisfied" if satisfied else ("Average" if average else "Not Satisfied")
                    log_feedback(
                        query=last_query,
                        response=last_response,
                        rating=rating,
                        exec_time=st.session_state.exec_time if st.session_state.exec_time else 0.0
                    )
        else:
            st.caption("Run a query to enable the feedback form.")

    # --- [NEW] Example Queries in an Expander ---
    with st.expander("💡 Quick Test Queries"):
        example_queries = {
            "Hybrid Query (Synthesis)": "I have a faded road sign for a hospital, and I'm also worried about cars speeding in that area.",
            "Extractor Query (Fact Check)": "What is the radius for a speed hump at 25 km/h?",
            "Compliance Query (Comparison)": "I found a traffic lane line that is 150 mm wide. Is it compliant?",
            "Constraint Query (Rule Check)": "I want to put a box marking at an un-signalised roundabout. What are the rules for this?",
        }
        cols = st.columns(len(example_queries))
        for i, (key, value) in enumerate(example_queries.items()):
            if cols[i].button(key, use_container_width=True, key=f"ex_{i}"):
                st.session_state.run_example_query_content = value
                st.rerun() # Rerun initiated by example button

    # --- RERUN LOGIC ---
    if "run_example_query_content" in st.session_state and st.session_state.run_example_query_content:
        prompt_to_process = st.session_state.run_example_query_content
        st.session_state.run_example_query_content = None
    elif user_prompt:
        prompt_to_process = user_prompt

    # Process the prompt if available
    if prompt_to_process:
        st.session_state.exec_time = None 
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        
        # Display user message *immediately*
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt_to_process)
        
        # Stream Assistant Response
        with chat_container:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                agent_thoughts = ""
                
                start_time = time.time()

                try:
                    f = io.StringIO()
                    with redirect_stdout(f):
                        with st.spinner("Analyzing standards and synthesizing answer..."):
                            for chunk in agent_chain.stream({"query": prompt_to_process}): 
                                full_response += chunk
                                message_placeholder.markdown(full_response + "▌")

                    end_time = time.time()
                    st.session_state.exec_time = end_time - start_time
                    message_placeholder.markdown(full_response)
                    agent_thoughts = f.getvalue()

                except Exception as e:
                    end_time = time.time()
                    st.session_state.exec_time = end_time - start_time
                    error_message = f"**Sorry, an unexpected internal error occurred.**\n\n*Error details: {e.__class__.__name__}: {str(e)}*"
                    message_placeholder.error(error_message)
                    st.toast("Internal RAG error occurred.", icon="❌")
                    full_response = error_message
                    agent_thoughts = f.getvalue() + "\n\n" + traceback.format_exc()

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        # [NEW] Store thoughts in session state to be displayed in the expander
        st.session_state.last_agent_thoughts = agent_thoughts
        st.rerun()


    # --- [NEW] Collapsible Section for Tall Content (Agent Thoughts & Filter Results) ---
    with st.expander("🔬 Agent Process & Filter Results", expanded=False):
        
        # --- Display Agent Thoughts ---
        if show_thoughts:
            st.subheader("🧠 Agent's Internal Process (RAG & Planning)")
            if "last_agent_thoughts" in st.session_state and st.session_state.last_agent_thoughts:
                st.code(st.session_state.last_agent_thoughts, language='text')
                st.session_state.last_agent_thoughts = None # Clear after showing
            else:
                st.caption("Run a query to see the agent's thoughts here.")
        
        st.markdown("---")

        # --- Display Engineer Mode Filter Results ---
        st.markdown("### 🛠️ Structured Search Results")
        if "filter_results" in st.session_state and st.session_state.filter_results:
            fr = st.session_state.filter_results
            
            if fr["error"]:
                st.error(fr["error"])
            elif fr["results"] is None:
                st.info("Click 'Search Database' in the sidebar to run a query.")
            elif not fr["results"]:
                st.warning(f"🚫 No matching interventions found for filters: `{fr['filters']}`")
            else:
                results_df = pd.DataFrame(fr['results'])
                st.success(f"✅ Found {len(fr['results'])} matching interventions for: **{fr['filters']}**")
                
                st.dataframe(
                    results_df[['S. No.', 'problem', 'category', 'type', 'code', 'clause']], 
                    use_container_width=True,
                    hide_index=True
                )

                with st.expander("View Full Details"):
                    for i, row in results_df.iterrows():
                        st.markdown(f"**{i+1}. {row['type']}** (Problem: {row['problem']}, Category: {row['category']})")
                        st.markdown(f"**Source:** `{row['code']}`, **Clause:** `{row['clause']}`")
                        st.markdown(row['data'])
                        st.markdown("---")
            
            # [NEW] Clear results after displaying them
            st.session_state.filter_results = None
        else:
            st.caption("Structured search results (from sidebar) will appear here.")


with tab2:
    st.header("📄 Knowledge Gap Log")
    st.markdown("Queries that the agent could not answer are logged here to identify missing data or poor retrieval performance.")
    log_content = load_knowledge_gaps()
    st.code(log_content, language="text")
    
    col_refresh, col_spacer = st.columns([0.2, 0.8])
    if col_refresh.button("Refresh Log", key="refresh_log_button"):
        st.cache_data.clear()
        st.rerun()


# import streamlit as st
# import sys
# import os
# import sqlalchemy
# import pandas as pd
# import time
# import datetime # <--- FIX: Explicitly importing datetime
# from contextlib import redirect_stdout
# import io
# import traceback # For detailed error printing
# from typing import Dict, Any

# # --- Add project root to path ---
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if os.path.basename(PROJECT_ROOT) == 'src':
#     PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)
# # --------------------------------
# try:
#     sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
#     from src import config, tools, agent
# except ImportError as e:
#     st.error(f"Error importing modules: {e}. Make sure 'src' directory exists and is in the correct path relative to app.py.")
#     st.stop()
# except Exception as e:
#     st.error(f"An unexpected error occurred during import: {e}")
#     st.stop()

# # --- Page Config ---
# st.set_page_config(
#     page_title="Sarathī-Chariot: Road Safety Expert Agent",
#     page_icon="🚧",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- HELPER FUNCTIONS ---
# @st.cache_resource
# def load_agent():
#     try:
#         chain = agent.create_supervisor_chain()
#         if hasattr(chain, 'llm'):
#             st.sidebar.success(f"LLM initialized: {config.OLLAMA_MODEL}")
#         return chain
#     except Exception as e:
#         st.error(f"Failed to load the agent: {e}. Check agent.py and dependencies.")
#         st.stop()
#         return None

# @st.cache_data
# def get_filter_options():
#     try:
#         if not os.path.exists(config.STRUCTURED_DB_PATH):
#             st.warning(f"Database not found at {config.STRUCTURED_DB_PATH}. Please run src/data_processor.py first.")
#             return [], [], []
#         engine = sqlalchemy.create_engine(f"sqlite:///{config.STRUCTURED_DB_PATH}")
#         with engine.connect() as conn:
#             problems = [row[0] for row in conn.execute(sqlalchemy.text(f"SELECT DISTINCT problem FROM {config.SQL_TABLE_NAME} ORDER BY problem"))]
#             categories = [row[0] for row in conn.execute(sqlalchemy.text(f"SELECT DISTINCT category FROM {config.SQL_TABLE_NAME} ORDER BY category"))]
#             types = [row[0] for row in conn.execute(sqlalchemy.text(f"SELECT DISTINCT type FROM {config.SQL_TABLE_NAME} ORDER BY type"))]
#         if not problems:
#             st.warning("Could not load filter options. Database might be empty or inaccessible.")
#             return [], [], []
#         return problems, categories, types
#     except Exception as e:
#         st.error(f"Error loading filter options from database: {e}")
#         return [], [], []

# def load_knowledge_gaps():
#     try:
#         if not os.path.exists(config.KNOWLEDGE_GAP_LOG_PATH):
#             return "No knowledge gaps logged yet. Queries that yield no results will be logged here."
#         with open(config.KNOWLEDGE_GAP_LOG_PATH, "r") as f:
#             content = f.read()
#             return content if content else "Log file is empty."
#     except Exception as e:
#         return f"Error loading log: {e}"

# def log_feedback(query: str, response: str, rating: str, exec_time: float):
#     """Logs user feedback to a dedicated file."""
#     feedback_path = os.path.join(PROJECT_ROOT, 'data', 'knowledge_gaps.txt')
#     try:
#         os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
#         with open(feedback_path, "a", encoding="utf-8") as f:
#             # FIX applied here: datetime is now imported and used correctly
#             timestamp = datetime.datetime.now().isoformat()
#             f.write(f"--- FEEDBACK --- [{timestamp} | Time: {exec_time:.2f}s | Rating: {rating}]\n")
#             f.write(f"Query: {query}\n")
#             f.write(f"Response Snippet: {response[:150]}...\n\n")
#         st.toast(f"Feedback logged: {rating}!", icon="👍")
#     except Exception as e:
#         st.warning(f"Could not log feedback: {e}")


# # --- Load Data & Agent ---
# agent_chain = load_agent()
# problems, categories, types = get_filter_options()


# # --- Custom CSS for Aesthetics ---
# st.markdown("""
# <style>
# /* Main container padding */
# .main .block-container {
#     padding-top: 1rem;
#     padding-right: 2rem;
#     padding-left: 2rem;
#     padding-bottom: 2rem;
# }
# /* Chat input appearance */
# .stChatInput > div > div {
#     border-radius: 0.5rem;
#     padding: 0.5rem;
# }
# /* Custom style for the engineer mode section title in the sidebar */
# #engineer-mode-filters {
#     color: #FF4B4B; /* Streamlit Red equivalent */
# }
# /* Button style improvements */
# .stButton>button {
#     border: 1px solid #FF4B4B;
#     border-radius: 0.5rem;
# }
# </style>
# """, unsafe_allow_html=True)


# # --- Sidebar: Engineer (Filter) Mode ---
# with st.sidebar:
#     st.markdown("<h2 id='engineer-mode-filters'>🔍 Engineer Mode (Filters)</h2>", unsafe_allow_html=True)
#     st.markdown("Use precise filters to query the structured IRC standards database.")

#     with st.container(border=True):
#         if not problems and not categories and not types:
#             st.error("Filter options failed to load. Cannot use Engineer Mode.")
#         else:
#             selected_problem = st.selectbox("🛑 Problem Type", options=["Any"] + problems, key="filter_problem")
#             selected_category = st.selectbox("🛣️ Asset Category", options=["Any"] + categories, key="filter_category")
#             selected_type = st.selectbox("🚧 Sign/Marking Type", options=["Any"] + types, key="filter_type")

#             if st.button("Search Database", use_container_width=True, key="filter_search"):
#                 filters = {}
#                 if selected_problem != "Any": filters["problem"] = selected_problem
#                 if selected_category != "Any": filters["category"] = selected_category
#                 if selected_type != "Any": filters["type"] = selected_type

#                 if not filters:
#                     st.warning("Please select at least one filter.")
#                 else:
#                     st.session_state.filter_results = {"filters": filters, "results": None, "error": None}
#                     with st.spinner("Querying structured database..."):
#                         try:
#                             st.session_state.filter_results["results"] = tools.structured_search(**filters)
#                         except Exception as e:
#                             st.session_state.filter_results["error"] = f"Error during structured search: {e}"

#     st.markdown("---")
#     st.info(f"Powered by Local RAG & **{config.OLLAMA_MODEL}** via Ollama.")


# # --- Main Page Layout ---
# st.markdown("<h1>Sarathī: Road Safety Expert Agent</h1>", unsafe_allow_html=True)
# st.subheader("Query the Indian Road Congress (IRC) standards with AI.")

# # Create tabs AFTER the title
# tab1, tab2 = st.tabs(["💬 Expert Chat", "📄 Knowledge Gap Log"])

# with tab1:
#     # --- Example Queries Section ---
#     st.markdown("#### Try a sample query or ask your own question below:")
#     example_queries = {
#         "Hybrid Query (Synthesis)": "I have a faded road sign for a hospital, and I'm also worried about cars speeding in that area.",
#         "Extractor Query (Fact Check)": "What is the radius for a speed hump at 25 km/h?",
#         "Compliance Query (Comparison)": "I found a traffic lane line that is 150 mm wide. Is it compliant?",
#         "Constraint Query (Rule Check)": "I want to put a box marking at an un-signalised roundabout. What are the rules for this?",
#     }
#     cols = st.columns(len(example_queries))
#     for i, (key, value) in enumerate(example_queries.items()):
#         if cols[i].button(key, use_container_width=True, key=f"ex_{i}"):
#             st.session_state.run_example_query_content = value
#             st.rerun()

#     st.markdown("---")
    
#     # --- Chat History & Input ---
#     chat_container = st.container(height=600, border=False) 

#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#         st.session_state.exec_time = None

#     # Display chat messages from history inside the container
#     with chat_container:
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#     # Display execution time if available
#     if st.session_state.exec_time:
#         st.info(f"⏱️ **Agent Processing Time:** {st.session_state.exec_time:.2f} seconds")
    
#     # NEW: Feedback Form
#     if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
#         st.markdown("---")
#         with st.form("feedback_form", clear_on_submit=True):
#             st.markdown("##### Did this answer satisfy your query?")
#             feedback_cols = st.columns(3)
            
#             # Emoji Selection
#             satisfied = feedback_cols[0].form_submit_button("Satisfied 😊")
#             average = feedback_cols[1].form_submit_button("Average 😐")
#             not_satisfied = feedback_cols[2].form_submit_button("Not Satisfied 😔")
            
#             # Hidden fields for logging
#             last_query = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else "N/A"
#             last_response = st.session_state.messages[-1]["content"] if st.session_state.messages else "N/A"
            
#             if satisfied or average or not_satisfied:
#                 rating = "Satisfied" if satisfied else ("Average" if average else "Not Satisfied")
                
#                 # Log the feedback
#                 log_feedback(
#                     query=last_query,
#                     response=last_response,
#                     rating=rating,
#                     exec_time=st.session_state.exec_time if st.session_state.exec_time else 0.0
#                 )


#     # Checkbox for Agent Thoughts (placed below the container for separation)
#     show_thoughts = st.checkbox("Show Agent Thoughts/Process (Detailed RAG Steps)", value=False, key="show_thoughts")

#     # --- Input Handling Logic ---
#     user_prompt = st.chat_input("What is your road safety question?", key="chat_input")
#     prompt_to_process = None

#     if "run_example_query_content" in st.session_state and st.session_state.run_example_query_content:
#         prompt_to_process = st.session_state.run_example_query_content
#         st.session_state.run_example_query_content = None
#     elif user_prompt:
#         prompt_to_process = user_prompt

#     # Process the prompt if available
#     if prompt_to_process:
#         st.session_state.exec_time = None 
        
#         # Add User message to history
#         st.session_state.messages.append({"role": "user", "content": prompt_to_process})
#         with chat_container:
#             with st.chat_message("user"):
#                 st.markdown(prompt_to_process)
        
#         # Stream Assistant Response
#         with chat_container:
#             with st.chat_message("assistant"):
#                 message_placeholder = st.empty()
#                 full_response = ""
#                 agent_thoughts = ""
                
#                 start_time = time.time()

#                 try:
#                     f = io.StringIO()
#                     with redirect_stdout(f):
#                         with st.spinner("Analyzing standards and synthesizing answer..."):
#                             for chunk in agent_chain.stream({"query": prompt_to_process}): 
#                                 full_response += chunk
#                                 message_placeholder.markdown(full_response + "▌")

#                     end_time = time.time()
#                     st.session_state.exec_time = end_time - start_time
                    
#                     message_placeholder.markdown(full_response)
#                     agent_thoughts = f.getvalue()

#                 except Exception as e:
#                     end_time = time.time()
#                     st.session_state.exec_time = end_time - start_time
                    
#                     error_message = f"**Sorry, an unexpected internal error occurred.**\n\n*Error details: {e.__class__.__name__}: {str(e)}*"
#                     message_placeholder.error(error_message)
#                     st.error("Detailed Error Traceback (See below the chat):")
#                     st.code(traceback.format_exc(), language='python')
#                     full_response = error_message
#                     agent_thoughts = f.getvalue() + "\n\n" + traceback.format_exc()

#         st.session_state.messages.append({"role": "assistant", "content": full_response})

#         if show_thoughts and agent_thoughts:
#             st.markdown("---")
#             st.subheader("🧠 Agent's Internal Process (RAG & Planning)")
#             st.code(agent_thoughts, language='text')

#         st.rerun()


# # --- Display Engineer Mode Filter Results ---
# if "filter_results" in st.session_state and st.session_state.filter_results:
#     st.markdown("---")
#     st.markdown("### 🛠️ Structured Search Results")
#     fr = st.session_state.filter_results
    
#     if fr["error"]:
#         st.error(fr["error"])
#     elif fr["results"] is None:
#         st.info("Click 'Search Database' in the sidebar to run a query.")
#     elif not fr["results"]:
#         st.info(f"🚫 No matching interventions found for filters: `{fr['filters']}`")
#     else:
#         results_df = pd.DataFrame(fr['results'])
#         st.success(f"✅ Found {len(fr['results'])} matching interventions for: **{fr['filters']}**")
        
#         st.dataframe(
#             results_df[['S. No.', 'problem', 'category', 'type', 'code', 'clause']], 
#             use_container_width=True,
#             hide_index=True
#         )

#         with st.expander("View Full Details"):
#             for i, row in results_df.iterrows():
#                 st.markdown(f"**{i+1}. {row['type']}** (Problem: {row['problem']}, Category: {row['category']})")
#                 st.markdown(f"**Source:** `{row['code']}`, **Clause:** `{row['clause']}`")
#                 st.markdown("---")
#                 st.markdown(row['data'])
#                 st.markdown("---")


# with tab2:
#     st.header("📄 Knowledge Gap Log")
#     st.markdown("Queries that the agent could not answer are logged here to identify missing data or poor retrieval performance.")
#     log_content = load_knowledge_gaps()
#     st.code(log_content, language="text")
    
#     col_refresh, col_spacer = st.columns([0.2, 0.8])
#     if col_refresh.button("Refresh Log", key="refresh_log_button"):
#         st.cache_data.clear()
#         st.rerun()
