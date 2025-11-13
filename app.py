import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import sqlalchemy
from contextlib import redirect_stdout
import io
import traceback
import html
import json
import time
from datetime import datetime

# --- 1. PROJECT SETUP AND IMPORTS ---
# Ensure 'src' is in the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
try:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
    from src import config, tools, agent
except Exception as e:
    st.error(f"Error importing modules: {e}. Make sure 'src' directory exists and is in the correct path relative to app.py.")
    st.stop()


# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SARATHI: Road Safety Expert Agent", 
    page_icon="🚦", 
    layout="wide", 
    initial_sidebar_state="auto"
)


# --- 3. CUSTOM STYLING (CSS) ---
st.markdown("""
<style>
/* 1. Global Font and Background */
@import url('https://fonts.googleapis.com/css2?family=Ubuntu:wght@400;500;700&display=swap');

html, body, [class*="stApp"] {
    font-family: 'Ubuntu', sans-serif;
    color: #172B4D; /* Deep blue/charcoal text color */
}
[data-testid="stAppViewContainer"] {
    background-color: #F8FAFC; /* Very light background for the whole page */
}

/* 2. Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #E6EEF5; /* Light blue/grey sidebar background */
    border-right: 2px solid #DCE3E9;
    box-shadow: 2px 0 5px rgba(0,0,0,0.05);
}

/* 3. Header/Title Styling */
h1 {
    color: #004D99; /* Deep blue title color */
    border-bottom: 2px solid #004D99; /* Underline the title */
    padding-bottom: 10px;
}
[data-testid="stCaption"] {
    font-size: 14px;
    color: #4A6572;
}

/* 4. Primary Button Styling (Example Queries, Search) */
.stButton>button[kind="primary"] {
    background-color: #007bff;
    border: 1px solid #007bff;
    color: white !important;
    font-weight: 500;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 123, 255, 0.2);
    transition: all 0.3s;
}
.stButton>button[kind="primary"]:hover {
    background-color: #0056b3;
    border-color: #0056b3;
}

/* 5. Secondary Button Styling (Example Queries) */
.stButton>button:not([kind="primary"]) {
    border: 1px solid #DCE3E9;
    background-color: white;
    color: #333;
    border-radius: 8px;
    transition: all 0.3s;
}
.stButton>button:not([kind="primary"]):hover {
    background-color: #f0f2f6;
    border-color: #C3CFD9;
}

/* 6. Info Box Accent Color */
[data-testid="stAlert"] > div:has(div.stAlert) {
    background-color: #E6EEF5;
    border-left: 5px solid #007bff;
}

</style>
""", unsafe_allow_html=True)


# --- 4. CHAT WINDOW RENDERING FUNCTION ---
def render_chat_window(messages, height_px=520):
    """
    Renders a single fixed-height, scrollable chat window with:
      - avatars
      - timestamps
      - assistant generation duration ("Generated in X s")
      - markdown rendering (marked), sanitization (DOMPurify) and code highlighting (highlight.js)
    """
    msgs_json = json.dumps(messages)

    full_html = f"""
    <link href="https://fonts.googleapis.com/css2?family=Ubuntu:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/default.min.css">
    
    <style>
        /* Chat Window Specific Styling (Refined Colors and Font) */
        :root {{
            --user-bg: #E6F3FF; /* Very light blue for user input */
            --user-text: #004D99; /* Deep blue text for user */
            --assistant-bg: #FFFFFF; /* White/Subtle off-white gradient for assistant */
            --assistant-text: #172B4D; /* Darker text for assistant */
            --muted: #6b7280;
            --bubble-radius: 18px;
            --main-font: 'Ubuntu', sans-serif; 
        }}
        
        .chat-frame {{
            border: 1px solid #DCE3E9;
            border-radius: 12px;
            padding: 12px;
            height: {height_px}px;
            overflow-y: auto;
            background: #FFFFFF; /* Solid white chat box */
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            font-family: var(--main-font); 
        }}

        .message-row {{ display:flex; gap:10px; margin:12px 0; align-items:flex-end; }}
        .message-row.user {{ justify-content:flex-end; }}
        .message-row.assistant {{ justify-content:flex-start; }}

        .avatar {{
            width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center;
            font-size:18px; flex: 0 0 36px;
            box-shadow: 0 1px 0 rgba(0,0,0,0.04);
        }}
        .avatar.user {{ background: #C8E1FF; }} /* Light blue avatar */
        .avatar.assistant {{ background: #FFD700; color: #172B4D; }} /* Yellow/Gold for Agent */

        .bubble {{
            max-width:78%;
            padding:12px 14px;
            border-radius: var(--bubble-radius);
            line-height:1.45;
            box-shadow: 0 2px 8px rgba(16,24,40,0.04);
            font-family: var(--main-font);
            font-size:14px;
            white-space:pre-wrap;
            word-wrap:break-word;
            overflow-wrap:anywhere;
            position:relative;
        }}

        .user-bubble {{
            background: var(--user-bg);
            color: var(--user-text);
            border-bottom-right-radius:6px;
            border: 1px solid #B0D7FF;
        }}
        .assistant-bubble {{
            background: var(--assistant-bg);
            color: var(--assistant-text);
            border-bottom-left-radius:6px;
            border: 1px solid #DCE3E9;
        }}

        .bubble .header {{ display:flex; align-items:center; gap:8px; margin-bottom:6px; }}
        .bubble .sender-name {{ font-weight:600; font-size:13px; }}
        .bubble .meta-badge {{ font-size:12px; color:var(--muted); background: rgba(0,0,0,0.02); padding:2px 8px; border-radius:999px; }}

        .bubble h1, .bubble h2, .bubble h3 {{ margin: 6px 0; font-family: var(--main-font); }}
        .bubble p {{ margin: 6px 0; }}
        .bubble ul, .bubble ol {{ margin: 6px 0 6px 18px; }}
        .bubble code {{ background: rgba(27,31,35,0.06); padding: 2px 6px; border-radius:6px; }}
        .bubble pre {{ background: #172B4D; color:#e6edf3; padding:10px; border-radius:8px; overflow:auto; }} /* Dark code block */

        .bubble .footer {{
            margin-top:10px;
            display:flex;
            gap:8px;
            align-items:center;
            justify-content:flex-end;
            font-size:12px;
            color:var(--muted);
        }}
        .bubble .duration {{
            font-weight:600;
            color: #007bff; /* Primary blue for timing */
            background: rgba(0, 123, 255, 0.1);
            padding:2px 8px;
            border-radius:999px;
        }}

        .chat-frame::-webkit-scrollbar {{ width:10px; }}
        .chat-frame::-webkit-scrollbar-thumb {{ background: rgba(0,0,0,0.08); border-radius:8px; }}
    </style>

    <div id="chat-window" class="chat-frame"></div>

    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/2.4.0/purify.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>

    <script>
        const messages = {msgs_json};
        const container = document.getElementById('chat-window');
        container.innerHTML = '';

        messages.forEach(m => {{
            const row = document.createElement('div');
            row.className = 'message-row ' + (m.role === 'user' ? 'user' : 'assistant');

            // avatar
            const avatar = document.createElement('div');
            avatar.className = 'avatar ' + (m.role === 'user' ? 'user' : 'assistant');
            avatar.textContent = m.avatar_emoji || (m.role === 'user' ? '👤' : '🤖');

            // bubble
            const bubble = document.createElement('div');
            bubble.className = 'bubble ' + (m.role === 'user' ? 'user-bubble' : 'assistant-bubble');

            // header (name)
            const header = document.createElement('div');
            header.className = 'header';
            const senderName = document.createElement('div');
            senderName.className = 'sender-name';
            senderName.textContent = m.sender_name || (m.role === 'user' ? 'You' : 'Agent');
            header.appendChild(senderName);
            if (m.role === 'assistant' && m.model_label) {{
                const meta = document.createElement('div');
                meta.className = 'meta-badge';
                meta.textContent = m.model_label;
                header.appendChild(meta);
            }}
            bubble.appendChild(header);

            // markdown -> html (sanitize)
            try {{
                const content_html = DOMPurify.sanitize(marked.parse(m.content || ''));
                const content_div = document.createElement('div');
                content_div.innerHTML = content_html;
                bubble.appendChild(content_div);
            }} catch (err) {{
                const fallback = document.createElement('div');
                fallback.textContent = m.content || '';
                bubble.appendChild(fallback);
            }}

            // footer
            const footer = document.createElement('div');
            footer.className = 'footer';
            const ts = document.createElement('div');
            ts.textContent = m.time || '';
            footer.appendChild(ts);
            if (m.role === 'assistant' && (m.duration !== null && m.duration !== undefined)) {{
                const dur = document.createElement('div');
                dur.className = 'duration';
                dur.textContent = `Generated in ${{m.duration.toFixed(2)}} s`;
                footer.appendChild(dur);
            }}
            bubble.appendChild(footer);

            // assemble row with deterministic order for correct alignment:
            if (m.role === 'assistant') {{
                row.appendChild(avatar);
                row.appendChild(bubble);
            }} else {{
                // user: bubble first, then avatar (so aligned right)
                row.appendChild(bubble);
                row.appendChild(avatar);
            }}
            container.appendChild(row);
        }});

        // highlight code blocks
        if (window.hljs) {{
            document.querySelectorAll('pre code').forEach((block) => {{
                try {{ hljs.highlightElement(block); }} catch(e) {{ /* ignore */ }}
            }});
        }}

        // auto-scroll to bottom smoothly
        container.scrollTo({{ top: container.scrollHeight, behavior: 'smooth' }});
    </script>
    """

    components.html(full_html, height=height_px + 20, scrolling=True)


# --- 5. CACHED HELPER FUNCTIONS ---
@st.cache_resource
def load_agent():
    try:
        return agent.create_supervisor_chain()
    except Exception as e:
        st.error(f"Failed to load the agent: {e}")
        return None

@st.cache_data
def get_filter_options():
    try:
        if not os.path.exists(config.STRUCTURED_DB_PATH):
            st.error(f"Database not found at {config.STRUCTURED_DB_PATH}. Please run src/data_processor.py first.")
            return [], [], []
        engine = sqlalchemy.create_engine(f"sqlite:///{config.STRUCTURED_DB_PATH}")
        with engine.connect() as conn:
            problems = [row[0] for row in conn.execute(sqlalchemy.text(f"SELECT DISTINCT problem FROM {config.SQL_TABLE_NAME} ORDER BY problem"))]
            categories = [row[0] for row in conn.execute(sqlalchemy.text(f"SELECT DISTINCT category FROM {config.SQL_TABLE_NAME} ORDER BY category"))]
            types = [row[0] for row in conn.execute(sqlalchemy.text(f"SELECT DISTINCT type FROM {config.SQL_TABLE_NAME} ORDER BY type"))]
        return problems, categories, types
    except Exception as e:
        st.error(f"Error loading filter options from database: {e}")
        return [], [], []

@st.cache_data
def load_knowledge_gaps():
    try:
        if not os.path.exists(config.KNOWLEDGE_GAP_LOG_PATH):
            return "No knowledge gaps logged yet. Queries that yield no results will be logged here."
        with open(config.KNOWLEDGE_GAP_LOG_PATH, "r") as f:
            content = f.read()
            return content if content else "Log file is empty."
    except Exception as e:
        return f"Error loading log: {e}"

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- 6. LOAD RESOURCES ---
agent_chain = load_agent()
if agent_chain is None:
    st.stop()

problems, categories, types = get_filter_options()


# --- 7. SIDEBAR LAYOUT ---
with st.sidebar:
    st.title("🚦 About this App")
    st.info("This app uses a local LLM (via Ollama) and a curated database based on IRC standards for the 'Road Safety Intervention GPT' contest.")
    st.markdown("---")
    st.markdown("Designed to help road safety experts and engineers find suitable interventions for specific issues.")
    
    st.markdown("---") # Separator
    
    # --- FILTERS MOVED TO SIDEBAR ---
    st.title("🛠 Engineer Mode") 
    st.markdown("Use filters to query the database. Results will appear in the 'Engineer Mode (Search)' tab.")

    if not problems and not categories and not types:
        st.error("Filter options failed to load. Cannot use Engineer Mode.")
    else:
        selected_problem = st.selectbox("Problem", options=["Any"] + problems, key="filter_problem")
        selected_category = st.selectbox("Category", options=["Any"] + categories, key="filter_category")
        selected_type = st.selectbox("Type", options=["Any"] + types, key="filter_type")
        
        # This button click will set session_state and tab2 will read from it
        if st.button("Search with Filters", use_container_width=True, key="filter_search", type="primary"):
            filters = {}
            if selected_problem != "Any": filters["problem"] = selected_problem
            if selected_category != "Any": filters["category"] = selected_category
            if selected_type != "Any": filters["type"] = selected_type
            
            if not filters:
                st.warning("Please select at least one filter.")
                st.session_state.filter_results = None # Clear old results
            else:
                st.session_state.filter_results = {"filters": filters, "results": None, "error": None}
                with st.spinner("Querying structured database..."):
                    try:
                        st.session_state.filter_results["results"] = tools.structured_search(**filters)
                    except Exception as e:
                        st.session_state.filter_results["error"] = f"Error during structured search: {e}"


# --- 8. MAIN PAGE LAYOUT ---
st.title("SARATHI: Road Safety Expert Agent")
st.caption("Your AI assistant for IRC standards and road safety interventions, built for the 'Road Safety Intervention GPT' contest.")

tab1, tab2 = st.tabs(["💬 Expert Chat", "📄 Knowledge Gap Log"])


# --- TAB 1: EXPERT CHAT ---
with tab1:
    st.markdown("Ask about road safety interventions (IRC standards).")

    example_queries = {
        "Hybrid Query": "I have a faded road sign for a hospital, and I'm also worried about cars speeding in that area.",
        "Extractor Query": "What is the radius for a speed hump at 25 km/h?",
        "Concept Query": "What are Bi-Directional hazard markers used for?",
        "Knowledge Gap Query": "What are the rules for red light cameras?"
    }

    st.markdown("##### Try an example query:")
    ex_cols = st.columns(len(example_queries))
    for i, (label, q) in enumerate(example_queries.items()):
        if ex_cols[i].button(label, key=f"ex_q_{i}", use_container_width=True):
            st.session_state.pending_example = q

    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Process example query if one was clicked
    if st.session_state.get("pending_example"):
        prompt_to_process = st.session_state.pop("pending_example")
        
        st.session_state.messages.append({
            "role": "user", "content": prompt_to_process, "time": now_str(),
            "duration": None, "sender_name": "You", "avatar_emoji": "👤"
        })

        start_t = time.time()
        full_response, agent_thoughts = "", ""
        try:
            f = io.StringIO()
            with redirect_stdout(f):
                with st.spinner("Thinking..."):
                    for chunk in agent_chain.stream(prompt_to_process):
                        full_response += chunk
            agent_thoughts = f.getvalue()
        except Exception as e:
            full_response = f"Sorry, an error occurred: {e}"
            st.error(full_response)
            st.code(traceback.format_exc())
        duration = time.time() - start_t

        st.session_state.messages.append({
            "role": "assistant", "content": full_response, "time": now_str(),
            "duration": duration, "sender_name": "RoadSafety-LLM",
            "avatar_emoji": "🤖", "model_label": "local-ollama"
        })
        st.session_state.last_agent_thoughts = agent_thoughts

    # Render the chat window
    render_chat_window(st.session_state.messages, height_px=520)

    # Process new user input from chat_input
    if prompt := st.chat_input("Ask your road safety question..."):
        prompt_to_process = prompt.strip()

        st.session_state.messages.append({
            "role": "user", "content": prompt_to_process, "time": now_str(),
            "duration": None, "sender_name": "You", "avatar_emoji": "👤"
        })

        # Call agent and measure duration
        start_t = time.time()
        full_response, agent_thoughts = "", ""
        try:
            f = io.StringIO()
            with redirect_stdout(f):
                with st.spinner("Thinking..."):
                    for chunk in agent_chain.stream(prompt_to_process):
                        full_response += chunk
            agent_thoughts = f.getvalue()
        except Exception as e:
            full_response = f"Sorry, an error occurred: {e}"
            st.error(full_response)
            st.code(traceback.format_exc())
        duration = time.time() - start_t

        # Append assistant reply
        st.session_state.messages.append({
            "role": "assistant", "content": full_response, "time": now_str(),
            "duration": duration, "sender_name": "RoadSafety-LLM",
            "avatar_emoji": "🤖", "model_label": "local-ollama"
        })
        
        st.session_state.last_agent_thoughts = agent_thoughts
        st.rerun()

    # Checkbox for agent thoughts
    if st.checkbox("Show Agent Thoughts/Process (from last message)", value=False, key="show_thoughts"):
        st.markdown("---")
        st.subheader("🕵 Agent's Internal Process")
        if "last_agent_thoughts" in st.session_state and st.session_state.last_agent_thoughts:
            st.code(st.session_state.last_agent_thoughts, language="text")
        else:
            st.info("No agent thoughts captured yet. Send a message to see the process.")


# --- TAB 2: ENGINEER MODE (RESULTS) ---
# with tab2:
#     st.header("🛠 Engineer Mode: Structured Search Results")
#     st.markdown("Use the filters in the **sidebar** to query the structured database. Results will appear below.")
#     st.markdown("---")
    
#     # Show filter results
#     st.subheader("Filter Results")
#     if "filter_results" in st.session_state and st.session_state.filter_results:
#         fr = st.session_state.filter_results
#         if fr["error"]:
#             st.error(fr["error"])
#         elif fr["results"] is None:
#              st.info("Click 'Search with Filters' in the sidebar to see results here.")
#         elif not fr["results"]:
#              st.info(f"No matching interventions found for filters: {fr['filters']}")
#         else:
#              st.success(f"Found {len(fr['results'])} interventions for filters: {fr['filters']}")
#              for i, res in enumerate(fr['results']):
#                  with st.expander(f"**{i+1}. {res.get('type', 'N/A')}** (Problem: {res.get('problem', 'N/A')})"):
#                      st.markdown(f"*Source:* `{res.get('code', 'N/A')}`, *Clause:* `{res.get('clause', 'N/A')}`")
#                      st.markdown("---")
#                      st.markdown(res.get('data', 'No description available.'))
#     else:
#         st.info("Click 'Search with Filters' in the sidebar to see results here.")


# --- TAB 3: KNOWLEDGE GAP LOG ---
with tab2:
    st.header("📄 Knowledge Gap Log")
    st.markdown("Queries returning no results from the vector database are logged here.")
    log_content = load_knowledge_gaps()
    st.code(log_content, language="text")
    if st.button("Refresh Log"):
        st.cache_data.clear()
        st.rerun()
# import streamlit as st
# import streamlit.components.v1 as components
# import sys
# import os
# import sqlalchemy
# from contextlib import redirect_stdout
# import io
# import traceback
# import html
# import json
# import time
# from datetime import datetime

# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)
# try:
#     sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
#     from src import config, tools, agent
# except Exception as e:



#     st.error(f"Error importing modules: {e}. Make sure 'src' directory exists and is in the correct path relative to app.py.")
#     st.stop()



# st.set_page_config(
#     page_title="SARATHI: Road Safety Expert Agent", 
#     page_icon="🚦", 
#     layout="wide", 
#     initial_sidebar_state="auto"
# )



# st.markdown("""
# <style>
# /* 1. Global Font and Background */
# @import url('https://fonts.googleapis.com/css2?family=Ubuntu:wght@400;500;700&display=swap');

# html, body, [class*="stApp"] {
#     font-family: 'Ubuntu', sans-serif;
#     color: #172B4D; /* Deep blue/charcoal text color */
# }
# [data-testid="stAppViewContainer"] {
#     background-color: #F8FAFC; /* Very light background for the whole page */
# }

# /* 2. Sidebar Styling */
# [data-testid="stSidebar"] {
#     background-color: #E6EEF5; /* Light blue/grey sidebar background */
#     border-right: 2px solid #DCE3E9;
#     box-shadow: 2px 0 5px rgba(0,0,0,0.05);
# }

# /* 3. Header/Title Styling */
# h1 {
#     color: #004D99; /* Deep blue title color */
#     border-bottom: 2px solid #004D99; /* Underline the title */
#     padding-bottom: 10px;
# }
# [data-testid="stCaption"] {
#     font-size: 14px;
#     color: #4A6572;
# }

# /* 4. Primary Button Styling (Example Queries, Search) */
# /* This targets the 'primary' type button, giving it the deep blue primary color */
# .stButton>button[kind="primary"] {
#     background-color: #007bff;
#     border: 1px solid #007bff;
#     color: white !important;
#     font-weight: 500;
#     border-radius: 8px;
#     box-shadow: 0 2px 4px rgba(0, 123, 255, 0.2);
#     transition: all 0.3s;
# }
# .stButton>button[kind="primary"]:hover {
#     background-color: #0056b3;
#     border-color: #0056b3;
# }

# /* 5. Secondary Button Styling (Example Queries) */
# .stButton>button:not([kind="primary"]) {
#     border: 1px solid #DCE3E9;
#     background-color: white;
#     color: #333;
#     border-radius: 8px;
#     transition: all 0.3s;
# }
# .stButton>button:not([kind="primary"]):hover {
#     background-color: #f0f2f6;
#     border-color: #C3CFD9;
# }

# /* 6. Info Box Accent Color */
# [data-testid="stAlert"] > div:has(div.stAlert) {
#     background-color: #E6EEF5;
#     border-left: 5px solid #007bff;
# }

# </style>
# """, unsafe_allow_html=True)


# def render_chat_window(messages, height_px=520):
#     """
#     Renders a single fixed-height, scrollable chat window with:
#       - avatars
#       - timestamps
#       - assistant generation duration ("Generated in X s")
#       - markdown rendering (marked), sanitization (DOMPurify) and code highlighting (highlight.js)
#     """
#     msgs_json = json.dumps(messages)

#     full_html = f"""
#     <link href="https://fonts.googleapis.com/css2?family=Ubuntu:wght@400;500;700&display=swap" rel="stylesheet">
#     <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/default.min.css">
    
#     <style>
#       /* Chat Window Specific Styling (Refined Colors and Font) */
#       :root {{
#         --user-bg: #E6F3FF; /* Very light blue for user input */
#         --user-text: #004D99; /* Deep blue text for user */
#         --assistant-bg: #FFFFFF; /* White/Subtle off-white gradient for assistant */
#         --assistant-text: #172B4D; /* Darker text for assistant */
#         --muted: #6b7280;
#         --bubble-radius: 18px;
#         --main-font: 'Ubuntu', sans-serif; 
#       }}
      
#       .chat-frame {{
#         border: 1px solid #DCE3E9;
#         border-radius: 12px;
#         padding: 12px;
#         height: {height_px}px;
#         overflow-y: auto;
#         background: #FFFFFF; /* Solid white chat box */
#         box-shadow: 0 4px 12px rgba(0,0,0,0.05);
#         font-family: var(--main-font); 
#       }}

#       .message-row {{ display:flex; gap:10px; margin:12px 0; align-items:flex-end; }}
#       .message-row.user {{ justify-content:flex-end; }}
#       .message-row.assistant {{ justify-content:flex-start; }}

#       .avatar {{
#         width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center;
#         font-size:18px; flex: 0 0 36px;
#         box-shadow: 0 1px 0 rgba(0,0,0,0.04);
#       }}
#       .avatar.user {{ background: #C8E1FF; }} /* Light blue avatar */
#       .avatar.assistant {{ background: #FFD700; color: #172B4D; }} /* Yellow/Gold for Agent */

#       .bubble {{
#         max-width:78%;
#         padding:12px 14px;
#         border-radius: var(--bubble-radius);
#         line-height:1.45;
#         box-shadow: 0 2px 8px rgba(16,24,40,0.04);
#         font-family: var(--main-font);
#         font-size:14px;
#         white-space:pre-wrap;
#         word-wrap:break-word;
#         overflow-wrap:anywhere;
#         position:relative;
#       }}

#       .user-bubble {{
#         background: var(--user-bg);
#         color: var(--user-text);
#         border-bottom-right-radius:6px;
#         border: 1px solid #B0D7FF;
#       }}
#       .assistant-bubble {{
#         background: var(--assistant-bg);
#         color: var(--assistant-text);
#         border-bottom-left-radius:6px;
#         border: 1px solid #DCE3E9;
#       }}

#       .bubble .header {{ display:flex; align-items:center; gap:8px; margin-bottom:6px; }}
#       .bubble .sender-name {{ font-weight:600; font-size:13px; }}
#       .bubble .meta-badge {{ font-size:12px; color:var(--muted); background: rgba(0,0,0,0.02); padding:2px 8px; border-radius:999px; }}

#       .bubble h1, .bubble h2, .bubble h3 {{ margin: 6px 0; font-family: var(--main-font); }}
#       .bubble p {{ margin: 6px 0; }}
#       .bubble ul, .bubble ol {{ margin: 6px 0 6px 18px; }}
#       .bubble code {{ background: rgba(27,31,35,0.06); padding: 2px 6px; border-radius:6px; }}
#       .bubble pre {{ background: #172B4D; color:#e6edf3; padding:10px; border-radius:8px; overflow:auto; }} /* Dark code block */

#       .bubble .footer {{
#         margin-top:10px;
#         display:flex;
#         gap:8px;
#         align-items:center;
#         justify-content:flex-end;
#         font-size:12px;
#         color:var(--muted);
#       }}
#       .bubble .duration {{
#         font-weight:600;
#         color: #007bff; /* Primary blue for timing */
#         background: rgba(0, 123, 255, 0.1);
#         padding:2px 8px;
#         border-radius:999px;
#       }}

#       .chat-frame::-webkit-scrollbar {{ width:10px; }}
#       .chat-frame::-webkit-scrollbar-thumb {{ background: rgba(0,0,0,0.08); border-radius:8px; }}
#     </style>

#     <div id="chat-window" class="chat-frame"></div>

#     <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
#     <script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/2.4.0/purify.min.js"></script>
#     <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>

#     <script>
#       const messages = {msgs_json};
#       const container = document.getElementById('chat-window');
#       container.innerHTML = '';

#       messages.forEach(m => {{
#         const row = document.createElement('div');
#         row.className = 'message-row ' + (m.role === 'user' ? 'user' : 'assistant');

#         // avatar
#         const avatar = document.createElement('div');
#         avatar.className = 'avatar ' + (m.role === 'user' ? 'user' : 'assistant');
#         avatar.textContent = m.avatar_emoji || (m.role === 'user' ? '👤' : '🤖');

#         // bubble
#         const bubble = document.createElement('div');
#         bubble.className = 'bubble ' + (m.role === 'user' ? 'user-bubble' : 'assistant-bubble');

#         // header (name)
#         const header = document.createElement('div');
#         header.className = 'header';
#         const senderName = document.createElement('div');
#         senderName.className = 'sender-name';
#         senderName.textContent = m.sender_name || (m.role === 'user' ? 'You' : 'Agent');
#         header.appendChild(senderName);
#         if (m.role === 'assistant' && m.model_label) {{
#           const meta = document.createElement('div');
#           meta.className = 'meta-badge';
#           meta.textContent = m.model_label;
#           header.appendChild(meta);
#         }}
#         bubble.appendChild(header);

#         // markdown -> html (sanitize)
#         try {{
#           const content_html = DOMPurify.sanitize(marked.parse(m.content || ''));
#           const content_div = document.createElement('div');
#           content_div.innerHTML = content_html;
#           bubble.appendChild(content_div);
#         }} catch (err) {{
#           const fallback = document.createElement('div');
#           fallback.textContent = m.content || '';
#           bubble.appendChild(fallback);
#         }}

#         // footer
#         const footer = document.createElement('div');
#         footer.className = 'footer';
#         const ts = document.createElement('div');
#         ts.textContent = m.time || '';
#         footer.appendChild(ts);
#         if (m.role === 'assistant' && (m.duration !== null && m.duration !== undefined)) {{
#           const dur = document.createElement('div');
#           dur.className = 'duration';
#           dur.textContent = `Generated in ${{m.duration.toFixed(2)}} s`;
#           footer.appendChild(dur);
#         }}
#         bubble.appendChild(footer);

#         // assemble row with deterministic order for correct alignment:
#         if (m.role === 'assistant') {{
#           row.appendChild(avatar);
#           row.appendChild(bubble);
#         }} else {{
#           // user: bubble first, then avatar (so aligned right)
#           row.appendChild(bubble);
#           row.appendChild(avatar);
#         }}
#         container.appendChild(row);
#       }});

#       // highlight code blocks
#       if (window.hljs) {{
#         document.querySelectorAll('pre code').forEach((block) => {{
#           try {{ hljs.highlightElement(block); }} catch(e) {{ /* ignore */ }}
#         }});
#       }}

#       // auto-scroll to bottom smoothly
#       container.scrollTo({{ top: container.scrollHeight, behavior: 'smooth' }});
#     </script>
#     """

#     components.html(full_html, height=height_px + 20, scrolling=True)



# @st.cache_resource
# def load_agent():
#     try:
#         return agent.create_supervisor_chain()
#     except Exception as e:
#         st.error(f"Failed to load the agent: {e}")
#         return None

# @st.cache_data
# def get_filter_options():
#     try:
#         if not os.path.exists(config.STRUCTURED_DB_PATH):
#             st.error(f"Database not found at {config.STRUCTURED_DB_PATH}. Please run src/data_processor.py first.")
#             return [], [], []
#         engine = sqlalchemy.create_engine(f"sqlite:///{config.STRUCTURED_DB_PATH}")
#         with engine.connect() as conn:
#             problems = [row[0] for row in conn.execute(sqlalchemy.text(f"SELECT DISTINCT problem FROM {config.SQL_TABLE_NAME} ORDER BY problem"))]
#             categories = [row[0] for row in conn.execute(sqlalchemy.text(f"SELECT DISTINCT category FROM {config.SQL_TABLE_NAME} ORDER BY category"))]
#             types = [row[0] for row in conn.execute(sqlalchemy.text(f"SELECT DISTINCT type FROM {config.SQL_TABLE_NAME} ORDER BY type"))]
#         return problems, categories, types
#     except Exception as e:
#         st.error(f"Error loading filter options from database: {e}")
#         return [], [], []

# @st.cache_data
# def load_knowledge_gaps():
#     try:
#         if not os.path.exists(config.KNOWLEDGE_GAP_LOG_PATH):
#             return "No knowledge gaps logged yet. Queries that yield no results will be logged here."
#         with open(config.KNOWLEDGE_GAP_LOG_PATH, "r") as f:
#             content = f.read()
#             return content if content else "Log file is empty."
#     except Exception as e:
#         return f"Error loading log: {e}"

# agent_chain = load_agent()
# if agent_chain is None:
#     st.stop()

# problems, categories, types = get_filter_options()

# #Sidebar: About
# with st.sidebar:
#     st.title("🚦 About this App")
#     st.info("This app uses a local LLM (via Ollama) and a curated database based on IRC standards for the 'Road Safety Intervention GPT' contest.")
#     st.markdown("---")
#     st.markdown("Designed to help road safety experts and engineers find suitable interventions for specific issues.")


# st.title("SARATHI: Road Safety Expert Agent")
# st.caption("Your AI assistant for IRC standards and road safety interventions, built for the 'Road Safety Intervention GPT' contest.")


# tab1, tab2, tab3 = st.tabs(["💬 Expert Chat", "🛠 Engineer Mode (Search)", "📄 Knowledge Gap Log"])


# def now_str():
#     return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# with tab1:
#     st.markdown("Ask about road safety interventions (IRC standards).")

#     example_queries = {
#         "Hybrid Query": "I have a faded road sign for a hospital, and I'm also worried about cars speeding in that area.",
#         "Extractor Query": "What is the radius for a speed hump at 25 km/h?",
#         "Concept Query": "What are Bi-Directional hazard markers used for?",
#         "Knowledge Gap Query": "What are the rules for red light cameras?"
#     }

#     st.markdown("##### Try an example query:")
#     ex_cols = st.columns(len(example_queries))
#     for i, (label, q) in enumerate(example_queries.items()):
#         if ex_cols[i].button(label, key=f"ex_q_{i}", use_container_width=True):

#             st.session_state.pending_example = q

#     st.markdown("---")


#     if "messages" not in st.session_state:
#         st.session_state.messages = []


#     if st.session_state.get("pending_example"):
#         prompt_to_process = st.session_state.pop("pending_example")
        
#         st.session_state.messages.append({
#             "role": "user", "content": prompt_to_process, "time": now_str(),
#             "duration": None, "sender_name": "You", "avatar_emoji": "👤"
#         })

#         start_t = time.time()
#         full_response, agent_thoughts = "", ""
#         try:
#             f = io.StringIO()
#             with redirect_stdout(f):
#                 with st.spinner("Thinking..."):
#                     for chunk in agent_chain.stream(prompt_to_process):
#                         full_response += chunk
#             agent_thoughts = f.getvalue()
#         except Exception as e:
#             full_response = f"Sorry, an error occurred: {e}"
#             st.error(full_response)
#             st.code(traceback.format_exc())
#         duration = time.time() - start_t

#         st.session_state.messages.append({
#             "role": "assistant", "content": full_response, "time": now_str(),
#             "duration": duration, "sender_name": "RoadSafety-LLM",
#             "avatar_emoji": "🤖", "model_label": "local-ollama"
#         })
#         # Store thoughts for the checkbox
#         st.session_state.last_agent_thoughts = agent_thoughts



#     render_chat_window(st.session_state.messages, height_px=520)


#     if prompt := st.chat_input("Ask your road safety question..."):
#         prompt_to_process = prompt.strip()

#         st.session_state.messages.append({
#             "role": "user", "content": prompt_to_process, "time": now_str(),
#             "duration": None, "sender_name": "You", "avatar_emoji": "👤"
#         })

#         # Call agent and measure duration
#         start_t = time.time()
#         full_response, agent_thoughts = "", ""
#         try:
#             f = io.StringIO()
#             with redirect_stdout(f):
#                 # The spinner will appear globally while processing
#                 with st.spinner("Thinking..."):
#                     for chunk in agent_chain.stream(prompt_to_process):
#                         full_response += chunk
#             agent_thoughts = f.getvalue()
#         except Exception as e:
#             full_response = f"Sorry, an error occurred: {e}"
#             st.error(full_response)
#             st.code(traceback.format_exc())
#         duration = time.time() - start_t

#         # Append assistant reply
#         st.session_state.messages.append({
#             "role": "assistant", "content": full_response, "time": now_str(),
#             "duration": duration, "sender_name": "RoadSafety-LLM",
#             "avatar_emoji": "🤖", "model_label": "local-ollama"
#         })
        
#         st.session_state.last_agent_thoughts = agent_thoughts

#         st.rerun()


#     if st.checkbox("Show Agent Thoughts/Process (from last message)", value=False, key="show_thoughts"):
#         st.markdown("---")
#         st.subheader("🕵 Agent's Internal Process")
#         if "last_agent_thoughts" in st.session_state and st.session_state.last_agent_thoughts:
#             st.code(st.session_state.last_agent_thoughts, language="text")
#         else:
#             st.info("No agent thoughts captured yet. Send a message to see the process.")


# # Tab 2: Engineer Mode (Filters) 
# with tab2:
#     st.header("🛠 Engineer Mode: Structured Search")
#     st.markdown("Use precise filters to query the structured database directly. Results will appear below.")

#     if not problems and not categories and not types:
#         st.error("Filter options failed to load. Cannot use Engineer Mode.")
#     else:
#         selected_problem = st.selectbox("Problem", options=["Any"] + problems, key="filter_problem")
#         selected_category = st.selectbox("Category", options=["Any"] + categories, key="filter_category")
#         selected_type = st.selectbox("Type", options=["Any"] + types, key="filter_type")
        
#         if st.button("Search with Filters", use_container_width=True, key="filter_search", type="primary"):
#             filters = {}
#             if selected_problem != "Any": filters["problem"] = selected_problem
#             if selected_category != "Any": filters["category"] = selected_category
#             if selected_type != "Any": filters["type"] = selected_type
            
#             if not filters:
#                 st.warning("Please select at least one filter.")
#                 st.session_state.filter_results = None # Clear old results
#             else:
#                 st.session_state.filter_results = {"filters": filters, "results": None, "error": None}
#                 with st.spinner("Querying structured database..."):
#                     try:
#                         st.session_state.filter_results["results"] = tools.structured_search(**filters)
#                     except Exception as e:
#                         st.session_state.filter_results["error"] = f"Error during structured search: {e}"

#     # Show filter results
#     st.markdown("---")
#     st.subheader("Filter Results")
#     if "filter_results" in st.session_state and st.session_state.filter_results:
#         fr = st.session_state.filter_results
#         if fr["error"]:
#             st.error(fr["error"])
#         elif fr["results"] is None:
#              st.info("Click 'Search with Filters' to see results here.")
#         elif not fr["results"]:
#              st.info(f"No matching interventions found for filters: {fr['filters']}")
#         else:
#              st.success(f"Found {len(fr['results'])} interventions for filters: {fr['filters']}")
#              for i, res in enumerate(fr['results']):
#                  with st.expander(f"{i+1}. {res.get('type', 'N/A')}** (Problem: {res.get('problem', 'N/A')})"):
#                      st.markdown(f"*Source:* {res.get('code', 'N/A')}, *Clause:* {res.get('clause', 'N/A')}")
#                      st.markdown("---")
#                      st.markdown(res.get('data', 'No description available.'))
#     else:
#         st.info("Click 'Search with Filters' to see results here.")


# with tab3:
#     st.header("📄 Knowledge Gap Log")
#     st.markdown("Queries returning no results from the vector database are logged here.")
#     log_content = load_knowledge_gaps()
#     st.code(log_content, language="text")
#     if st.button("Refresh Log"):
#         st.cache_data.clear()
#         st.rerun()



