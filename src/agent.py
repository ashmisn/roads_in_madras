import sys
import os
import json
import datetime
import traceback
from typing import List, Dict, Any, Optional

# --- Add project root to path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# --------------------------------

try:
    
    from . import config, tools, prompts
except ImportError:
    
    try:
        from src import config, tools, prompts
    except ImportError:
        import config
        import tools
        import prompts

from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


try:
    
    llm = ChatOllama(model=config.OLLAMA_MODEL, base_url="http://localhost:11434")
    
    # --- MODIFICATION: This is no longer needed ---
    # reranker_llm = ChatOllama(model=config.OLLAMA_MODEL, base_url="http://localhost:11434", temperature=0.0)
    # --- END MODIFICATION ---
    
    print(f"LLM initialized with model: {config.OLLAMA_MODEL}")
    # print(f"Reranker LLM initialized with model: {config.OLLAMA_MODEL} (temp=0.0)")

except Exception as e:
    print(f"--- ERROR initializing LLM: {e} ---")
    print(f"Ensure Ollama is running and the model '{config.OLLAMA_MODEL}' is pulled.")
    sys.exit(1)

def log_knowledge_gap(query: str):
    """Logs a user query that returned no results to the knowledge gap file."""
    try:
        log_dir = os.path.dirname(config.KNOWLEDGE_GAP_LOG_PATH)
        os.makedirs(log_dir, exist_ok=True)
        with open(config.KNOWLEDGE_GAP_LOG_PATH, "a") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"[{timestamp}] {query}\n")
        print(f"Logged knowledge gap for query: {query}")
    except Exception as e:
        print(f"Error logging knowledge gap: {e}")
    
    return "[]" 



def execute_plan(input_dict: dict) -> dict:
    """
    Executes the multi-step plan from the Planner.
    (This function is unchanged)
    """
    agent_plan = input_dict.get('plan', None)
    query_string = str(input_dict.get('query', "Query unavailable"))

    if not agent_plan:
        print("Error: Agent Plan is missing.")
        return {"context": log_knowledge_gap(query_string), "query": query_string}

    tool_map = {
        "semantic_search": tools.semantic_search,
        "structured_search": tools.structured_search, # <-- UPDATED
    }

    INITIAL_K = 7 
    FINAL_TOP_K_PER_TOPIC = 3 
    print(f"\n--- Executing General Plan (Hybrid Search + RRF) ---")
    print(f"Planner Thought: {getattr(agent_plan, 'thought', 'N/A')}")

 
    structured_context = []

    plan_steps = getattr(agent_plan, 'plan', [])
    if not isinstance(plan_steps, list): plan_steps = []

    
    for tool_call in plan_steps:
        if not (hasattr(tool_call, 'tool_name') and hasattr(tool_call, 'arguments')):
             print(f"Warning: Invalid tool_call format: {tool_call}")
             continue
         
        tool_name = tool_call.tool_name
        arguments = tool_call.arguments
        tool_function = tool_map.get(tool_name)
         
        if not tool_function:
             print(f"Warning: Planner generated an invalid tool call: '{tool_name}'. Skipping.")
             continue

        all_results_for_this_topic = []
        topic_name = "general_search" # Fallback topic

        try:
            print(f"Calling (Planner): {tool_name} with args {arguments}")
            planned_results = tool_function(**arguments)
            all_results_for_this_topic.append({"planned": planned_results})

            if tool_name == "semantic_search":
                 sub_query = arguments.get('query')
                 if not sub_query:
                     print("Warning: Planner created a semantic_search call with no 'query' argument.")
                     continue
                 
                 topic_name = sub_query # Use the sub_query as the topic
                 print(f"--- Executing Sub-Query: '{sub_query}' ---")
                 
                 print(f"Calling (Keyword): keyword_search_sqlite for query '{sub_query}'")
                 keyword_results = tools.keyword_search_sqlite(query=sub_query, category_hint=None, n_results=INITIAL_K)
                 all_results_for_this_topic.append({"keyword": keyword_results})

            elif tool_name == "structured_search":
                 topic_name = f"structured search for {arguments.get('type', arguments.get('problem', 'data'))}"

        except Exception as e:
            print(f"--- Error executing planned tool {tool_name}: {e} ---")
            traceback.print_exc()


        print(f"Fusing {len(all_results_for_this_topic)} result lists for sub-query...")
         
        fused_results = []
        if hasattr(tools, 'rrf_fuse'):
            fused_results = tools.rrf_fuse(all_results_for_this_topic)
        else:
            print("Warning: 'rrf_fuse' function not found in tools.py. Falling back to simple combination.")
            for res_list in all_results_for_this_topic:
                fused_results.extend(res_list.values())


        top_docs_for_topic = fused_results[:FINAL_TOP_K_PER_TOPIC]
         

        for doc in top_docs_for_topic:
            structured_context.append({
                "retrieved_for_topic": topic_name, # Tag the doc with the topic
                "document": doc
            })

    print(f"\nSelected top {len(structured_context)} total documents across all topics to send to LLM.")
    print("----------------------------------------------------------\n")


    if not structured_context:
        context_str = log_knowledge_gap(query_string)
    else:
        try:
            
            context_str = json.dumps(structured_context, indent=2)
        except TypeError as e:
             print(f"Error serializing final results to JSON: {e}")
             context_str = "\n---\n".join([str(item) for item in structured_context])

    return {"context": context_str, "query": query_string}


def create_general_query_chain():
    """Builds the chain for handling general queries."""
    try:
        agent_plan_schema = prompts.AgentPlan
        final_answer_prompt_template = prompts.FINAL_ANSWER_PROMPT
    except AttributeError as e:
        print(f"--- ERROR: Missing Pydantic model or prompt in prompts module: {e} ---")
        sys.exit(1)

    plan_chain = (
        ChatPromptTemplate.from_template(prompts.PLANNER_SYSTEM_PROMPT)
        | llm.with_structured_output(agent_plan_schema)
    )
    synthesis_chain = (
        PromptTemplate.from_template(final_answer_prompt_template)
        | llm
        | StrOutputParser()
    )
    return RunnablePassthrough.assign(
        plan= RunnableLambda(lambda x: {"query": x["query"]}) | plan_chain
    ) | execute_plan | synthesis_chain


def create_extractor_chain():
    """
    Builds the chain for extracting specific facts using a new
    Hybrid Search -> Rerank -> Extract architecture.
    """
    try:
        extractor_output_schema = prompts.ExtractorOutput
        extractor_prompt_template = prompts.EXTRACTOR_PROMPT
        # --- MODIFICATION: These are no longer needed ---
        # reranker_output_schema = prompts.RerankedIDs
        # reranker_prompt_template = prompts.RERANKING_PROMPT
        # --- END MODIFICATION ---
    except AttributeError as e:
        print(f"--- ERROR: Missing Pydantic model or prompt in prompts module: {e} ---")
        sys.exit(1)

    def hybrid_search_for_extractor(input_dict: dict) -> dict:
        """
        Performs a full Hybrid Search (Semantic + Keyword + RRF)
        to get the best possible *candidate* documents for the reranker.
        (This function is unchanged)
        """
        query_string = str(input_dict.get("query", "")) # Ensure it's a string
        print(f"\n--- Executing Extractor (Step 1: Hybrid Search) ---")
         
        INITIAL_K = 5 
        semantic_results = []
        try:
            semantic_results = tools.semantic_search(query=query_string, n_results=INITIAL_K)
        except Exception as e:
             print(f"Error in extractor semantic search: {e}")
             
         
        keyword_results = []
        try:
            keyword_results = tools.keyword_search_sqlite(query=query_string, category_hint=None, n_results=INITIAL_K)
        except Exception as e:
            print(f"Error during keyword search execution: {e}")
             
         
        print(f"Fusing {len(semantic_results)} semantic and {len(keyword_results)} keyword results for extractor...")
        hybrid_results_list = [{"semantic": semantic_results, "keyword": keyword_results}]
        fused_results = []
        if hasattr(tools, 'rrf_fuse'):
            try:
                fused_results = tools.rrf_fuse(hybrid_results_list)
            except Exception as e:
                print(f"Error during RRF fusion: {e}")
                fused_results = semantic_results + keyword_results
        else:
            print("Warning: 'rrf_fuse' function not found in tools.py.")
            fused_results = semantic_results + keyword_results
         
        # --- Step 4: Select Top K Candidates ---
        final_candidates = fused_results[:INITIAL_K] # Send top 5 fused results to reranker
        print(f"Found {len(final_candidates)} total candidates for reranking.")

        if not final_candidates:
            log_knowledge_gap(query_string)
            return {"candidates": "[]", "query": query_string}
         
        return {"candidates": json.dumps(final_candidates, indent=2), "query": query_string}

    # --- MODIFICATION: Replaced entire function ---
    def rerank_documents(input_dict: dict) -> dict:
        """
        NEW Step 2: Takes candidate documents and uses the fast
        Cross-Encoder tool to find the single best document.
        """
        candidates_str = input_dict.get("candidates", "[]")
        query_string = input_dict.get("query", "")
         
        if candidates_str == "[]":
            print("No candidates to rerank.")
            return {"top_doc": None, "query": query_string} 
             
        print(f"--- Executing Extractor (Step 2: Cross-Encoder Reranking) ---")
         
        try:
            candidates_list = json.loads(candidates_str)
            
            # Call the new tool from tools.py
            # We ask for top_k=1 because the extractor only needs the *best* doc
            top_docs = tools.rerank_with_cross_encoder(
                query=query_string, 
                documents=candidates_list, 
                top_k=1
            )

            if not top_docs:
                 print("Reranker returned no relevant documents.")
                 return {"top_doc": None, "query": query_string}
            
            # The best document is the first one
            top_doc = top_docs[0]
            
            # Return it in the expected JSON format
            return {"top_doc": json.dumps([top_doc], indent=2), "query": query_string} 
             
        except Exception as e:
            print(f"--- Error during Cross-Encoder Reranking: {e} ---")
            traceback.print_exc()
            return {"top_doc": None, "query": query_string}
    # --- END MODIFICATION ---

    def format_extractor_output(output: Any) -> str: 
        """Helper function to format the Pydantic output from the extractor LLM."""
        if not isinstance(output, extractor_output_schema):
            print(f"Warning: Extractor LLM did not return expected Pydantic object. Output: {output}")
            return "Could not extract the answer due to an internal formatting error."

        if output.answer is None or "not found" in output.answer.lower():
             return "I could not find a specific answer for that in the database."
        source_str = ""
        if output.source_code and output.source_clause:
            source_str = f" (Source: {output.source_code}, Clause: {output.source_clause})"
        elif output.source_code:
             source_str = f" (Source: {output.source_code})"
        return f"The answer is: **{output.answer}**{source_str}"

    def route_extractor_logic(input_dict: dict):
        """
        NEW Step 3: Takes the single best document (or None) and routes
        to the final extraction LLM.
        (This function is unchanged)
        """
        top_doc_str = input_dict.get("top_doc", None)
        print(f"--- Executing Extractor (Step 3: Final Extraction) ---")
         
        if top_doc_str is None:
            return RunnableLambda(lambda x: "I could not find a specific answer for that in the database.")
        else:
           
            def remap_for_extractor(inputs):
                return {
                    "context": inputs.get("top_doc"),
                    "query": inputs.get("query")
                }

            extractor_prompt_obj = ChatPromptTemplate.from_template(extractor_prompt_template)
            extractor_llm_chain = (
                RunnableLambda(remap_for_extractor)
                | extractor_prompt_obj 
                | llm.with_structured_output(extractor_output_schema, include_raw=False) 
                | format_extractor_output
            )
            return extractor_llm_chain


    return (
 
        RunnableLambda(hybrid_search_for_extractor)

        | RunnableLambda(rerank_documents) # This now calls our new, fast function

        | RunnableLambda(route_extractor_logic)
    )



def create_supervisor_chain():
    """Builds the master chain that routes tasks."""
    try:
        query_router_schema = prompts.QueryRouter
        router_prompt_template = prompts.ROUTER_PROMPT
    except AttributeError as e:
        print(f"--- ERROR: Missing Pydantic model or prompt in prompts module: {e} ---")
        sys.exit(1)

    router_prompt_obj = ChatPromptTemplate.from_template(router_prompt_template)
    router_chain = router_prompt_obj | llm.with_structured_output(query_router_schema)

    print("Initializing chains...")
    general_chain = create_general_query_chain()
    extractor_chain = create_extractor_chain() 
    print("All chains initialized.")

    branch = RunnableBranch(
        (lambda x: x.get('task_type') == 'extraction_query', extractor_chain),

        general_chain 
    )

    return (

        {"task": {"query": RunnablePassthrough()} | router_chain,
         "query": RunnablePassthrough()} 
        | RunnableLambda(lambda x: {
            "task_type": getattr(x.get("task"), "task_type", "general_query"), 
            "query": x.get("query"), # Pass the original query string
            "thought": getattr(x.get("task"), "thought", "")
          })
        | RunnableLambda(lambda x: print(f"\n--- 🧠 Router Decision --- \nTask: {x['task_type']}\nThought: {x['thought']}\n{'-'*25}") or x) # Print router decision
        | branch
    )


if __name__ == "__main__":
    print("Road Safety Supervisor Agent Initialized.")
    print("This file is for logic. Run 'streamlit run app.py' to use the app.")

    if tools.sql_engine is None or tools.langchain_vector_store is None:
        print("\n--- WARNING: One or more tools failed to initialize. ---")
        print("--- Have you run 'python data_processor.py' first? ---")
        sys.exit(1)

    try:
        agent_chain = create_supervisor_chain()
    except Exception as e:
        print(f"--- FATAL ERROR during agent chain creation: {e} ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- Automatic Test Run ---
    print("\n--- INITIATING AUTOMATIC TEST RUN (FINAL VERSION) ---")
    

    test_queries = ["What is the minimum height requirement for a road sign above a pedestrian area, and how far apart should two successive signs be on an urban road?",
                    "I found a chevron sign that is 500 mm wide and 600 mm tall. Is this compliant with the standard dimensions for an 'Object Hazard (Left) Sign'?",
                    "My school's 'SCHOOL KEEP CLEAR' road marking is almost invisible due to fading. What is the required colour and size, and what must I do to fix this compliance problem?"]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"--- 🧪 EXECUTING TEST QUERY {i}/{len(test_queries)} ---")
        print(f"QUERY: {query}")
        print(f"{'-'*80}\n")
         
        try:
            print("--- Agent Answer ---")
            full_response = ""
            # We must pass the query as a string, as the chain expects string input
            for chunk in agent_chain.stream(query): 
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n--------------------")

        except KeyboardInterrupt: print("\nExiting..."); break
        except Exception as e:
             import traceback; print(f"\n--- Error During Query ---"); query_at_error = 'N/A'
             try: query_at_error = query
             except NameError: pass
             print(f"Query: {query_at_error}"); print(f"Error: {e}"); traceback.print_exc(); print(f"-------------------------")
     
    print(f"\n{'='*80}")
    print("✅ --- AUTOMATIC TEST RUN COMPLETE ---")
    print("Exiting.")


# import sys
# import os
# import json
# import datetime
# import traceback
# from typing import List, Dict, Any, Optional

# # --- Add project root to path ---
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)
# # --------------------------------

# try:
    
#     from . import config, tools, prompts
# except ImportError:
    
#     try:
#         from src import config, tools, prompts
#     except ImportError:
#         import config
#         import tools
#         import prompts

# from langchain_ollama.chat_models import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser


# try:
    
#     llm = ChatOllama(model=config.OLLAMA_MODEL, base_url="http://localhost:11434")
    
    
#     reranker_llm = ChatOllama(model=config.OLLAMA_MODEL, base_url="http://localhost:11434", temperature=0.0)
    
#     print(f"LLM initialized with model: {config.OLLAMA_MODEL}")
#     print(f"Reranker LLM initialized with model: {config.OLLAMA_MODEL} (temp=0.0)")
# except Exception as e:
#     print(f"--- ERROR initializing LLM: {e} ---")
#     print(f"Ensure Ollama is running and the model '{config.OLLAMA_MODEL}' is pulled.")
#     sys.exit(1)

# def log_knowledge_gap(query: str):
#     """Logs a user query that returned no results to the knowledge gap file."""
#     try:
#         log_dir = os.path.dirname(config.KNOWLEDGE_GAP_LOG_PATH)
#         os.makedirs(log_dir, exist_ok=True)
#         with open(config.KNOWLEDGE_GAP_LOG_PATH, "a") as f:
#             timestamp = datetime.datetime.now().isoformat()
#             f.write(f"[{timestamp}] {query}\n")
#         print(f"Logged knowledge gap for query: {query}")
#     except Exception as e:
#         print(f"Error logging knowledge gap: {e}")
    
#     return "[]" 



# def execute_plan(input_dict: dict) -> dict:
#     """
#     Executes the multi-step plan from the Planner.
#     It can now execute *both* semantic_search and structured_search calls.
#     It still performs a hybrid search (adding keyword search) for semantic_search plans
#     for maximum robustness.
#     It returns a structured JSON context mapping sub-queries (topics) to their documents.
#     """
#     agent_plan = input_dict.get('plan', None)
#     query_string = str(input_dict.get('query', "Query unavailable"))

#     if not agent_plan:
#         print("Error: Agent Plan is missing.")
#         return {"context": log_knowledge_gap(query_string), "query": query_string}

#     tool_map = {
#         "semantic_search": tools.semantic_search,
#         "structured_search": tools.structured_search, # <-- UPDATED
#     }

#     INITIAL_K = 7 
#     FINAL_TOP_K_PER_TOPIC = 3 
#     print(f"\n--- Executing General Plan (Hybrid Search + RRF) ---")
#     print(f"Planner Thought: {getattr(agent_plan, 'thought', 'N/A')}")

 
#     structured_context = []

#     plan_steps = getattr(agent_plan, 'plan', [])
#     if not isinstance(plan_steps, list): plan_steps = []

   
#     for tool_call in plan_steps:
#         if not (hasattr(tool_call, 'tool_name') and hasattr(tool_call, 'arguments')):
#              print(f"Warning: Invalid tool_call format: {tool_call}")
#              continue
        
#         tool_name = tool_call.tool_name
#         arguments = tool_call.arguments
#         tool_function = tool_map.get(tool_name)
        
#         if not tool_function:
#             print(f"Warning: Planner generated an invalid tool call: '{tool_name}'. Skipping.")
#             continue

#         all_results_for_this_topic = []
#         topic_name = "general_search" # Fallback topic

#         try:
#             print(f"Calling (Planner): {tool_name} with args {arguments}")
#             planned_results = tool_function(**arguments)
#             all_results_for_this_topic.append({"planned": planned_results})

#             if tool_name == "semantic_search":
#                 sub_query = arguments.get('query')
#                 if not sub_query:
#                     print("Warning: Planner created a semantic_search call with no 'query' argument.")
#                     continue
                
#                 topic_name = sub_query # Use the sub_query as the topic
#                 print(f"--- Executing Sub-Query: '{sub_query}' ---")
                
#                 print(f"Calling (Keyword): keyword_search_sqlite for query '{sub_query}'")
#                 keyword_results = tools.keyword_search_sqlite(query=sub_query, category_hint=None, n_results=INITIAL_K)
#                 all_results_for_this_topic.append({"keyword": keyword_results})

#             elif tool_name == "structured_search":
#                 topic_name = f"structured search for {arguments.get('type', arguments.get('problem', 'data'))}"

#         except Exception as e:
#             print(f"--- Error executing planned tool {tool_name}: {e} ---")
#             traceback.print_exc()


#         print(f"Fusing {len(all_results_for_this_topic)} result lists for sub-query...")
        
#         fused_results = []
#         if hasattr(tools, 'rrf_fuse'):
#             fused_results = tools.rrf_fuse(all_results_for_this_topic)
#         else:
#             print("Warning: 'rrf_fuse' function not found in tools.py. Falling back to simple combination.")
#             for res_list in all_results_for_this_topic:
#                 fused_results.extend(res_list.values())


#         top_docs_for_topic = fused_results[:FINAL_TOP_K_PER_TOPIC]
        

#         for doc in top_docs_for_topic:
#             structured_context.append({
#                 "retrieved_for_topic": topic_name, # Tag the doc with the topic
#                 "document": doc
#             })

#     print(f"\nSelected top {len(structured_context)} total documents across all topics to send to LLM.")
#     print("----------------------------------------------------------\n")


#     if not structured_context:
#         context_str = log_knowledge_gap(query_string)
#     else:
#         try:
            
#             context_str = json.dumps(structured_context, indent=2)
#         except TypeError as e:
#              print(f"Error serializing final results to JSON: {e}")
#              context_str = "\n---\n".join([str(item) for item in structured_context])

#     return {"context": context_str, "query": query_string}


# def create_general_query_chain():
#     """Builds the chain for handling general queries."""
#     try:
#         agent_plan_schema = prompts.AgentPlan
#         final_answer_prompt_template = prompts.FINAL_ANSWER_PROMPT
#     except AttributeError as e:
#         print(f"--- ERROR: Missing Pydantic model or prompt in prompts module: {e} ---")
#         sys.exit(1)

#     plan_chain = (
#         ChatPromptTemplate.from_template(prompts.PLANNER_SYSTEM_PROMPT)
#         | llm.with_structured_output(agent_plan_schema)
#     )
#     synthesis_chain = (
#         PromptTemplate.from_template(final_answer_prompt_template)
#         | llm
#         | StrOutputParser()
#     )
#     return RunnablePassthrough.assign(
#         plan= RunnableLambda(lambda x: {"query": x["query"]}) | plan_chain
#     ) | execute_plan | synthesis_chain


# def create_extractor_chain():
#     """
#     Builds the chain for extracting specific facts using a new
#     Hybrid Search -> Rerank -> Extract architecture.
#     """
#     try:
#         extractor_output_schema = prompts.ExtractorOutput
#         extractor_prompt_template = prompts.EXTRACTOR_PROMPT
#         reranker_output_schema = prompts.RerankedIDs
#         reranker_prompt_template = prompts.RERANKING_PROMPT
#     except AttributeError as e:
#         print(f"--- ERROR: Missing Pydantic model or prompt in prompts module: {e} ---")
#         sys.exit(1)

#     def hybrid_search_for_extractor(input_dict: dict) -> dict:
#         """
#         Performs a full Hybrid Search (Semantic + Keyword + RRF)
#         to get the best possible *candidate* documents for the reranker.
#         """
#         query_string = str(input_dict.get("query", "")) # Ensure it's a string
#         print(f"\n--- Executing Extractor (Step 1: Hybrid Search) ---")
        
#         INITIAL_K = 5 
#         semantic_results = []
#         try:
#             semantic_results = tools.semantic_search(query=query_string, n_results=INITIAL_K)
#         except Exception as e:
#              print(f"Error in extractor semantic search: {e}")
             
        
#         keyword_results = []
#         try:
#             keyword_results = tools.keyword_search_sqlite(query=query_string, category_hint=None, n_results=INITIAL_K)
#         except Exception as e:
#             print(f"Error during keyword search execution: {e}")
            
        
#         print(f"Fusing {len(semantic_results)} semantic and {len(keyword_results)} keyword results for extractor...")
#         hybrid_results_list = [{"semantic": semantic_results, "keyword": keyword_results}]
#         fused_results = []
#         if hasattr(tools, 'rrf_fuse'):
#             try:
#                 fused_results = tools.rrf_fuse(hybrid_results_list)
#             except Exception as e:
#                 print(f"Error during RRF fusion: {e}")
#                 fused_results = semantic_results + keyword_results
#         else:
#             print("Warning: 'rrf_fuse' function not found in tools.py.")
#             fused_results = semantic_results + keyword_results
        
#         # --- Step 4: Select Top K Candidates ---
#         final_candidates = fused_results[:INITIAL_K] # Send top 5 fused results to reranker
#         print(f"Found {len(final_candidates)} total candidates for reranking.")

#         if not final_candidates:
#             log_knowledge_gap(query_string)
#             return {"candidates": "[]", "query": query_string}
        
#         return {"candidates": json.dumps(final_candidates, indent=2), "query": query_string}

#     def rerank_documents(input_dict: dict) -> dict:
#         """
#         NEW Step 2: Takes candidate documents and uses an LLM to find the
#         single best document.
#         """
#         candidates_str = input_dict.get("candidates", "[]")
#         query_string = input_dict.get("query", "")
        
#         if candidates_str == "[]":
#             print("No candidates to rerank.")
#             return {"top_doc": None, "query": query_string} 
            
#         print(f"--- Executing Extractor (Step 2: Reranking) ---")
        
#         reranker_prompt_obj = ChatPromptTemplate.from_template(reranker_prompt_template)

#         reranker_chain = reranker_prompt_obj | reranker_llm.with_structured_output(reranker_output_schema, include_raw=False)
        
#         try:
#             reranked_output = reranker_chain.invoke({
#                 "query": query_string,
#                 "candidate_documents": candidates_str
#             })
            
#             if not reranked_output.reranked_ids:
#                 print("Reranker returned no relevant IDs.")
#                 return {"top_doc": None, "query": query_string}
            
#             top_id = reranked_output.reranked_ids[0] # Get the single best ID
#             print(f"Reranker selected S. No. {top_id} as best match.")
            

#             candidates_list = json.loads(candidates_str)
#             top_doc = next((doc for doc in candidates_list if int(doc.get("S. No.", -1)) == top_id), None)
            
#             if top_doc is None:
#                 print(f"Error: Reranker picked ID {top_id} but it was not in the candidate list.")
#                 return {"top_doc": None, "query": query_string}
            

#             return {"top_doc": json.dumps([top_doc], indent=2), "query": query_string} 
            
#         except Exception as e:
#             print(f"--- Error during Reranking: {e} ---")
#             traceback.print_exc()
#             return {"top_doc": None, "query": query_string}

#     def format_extractor_output(output: Any) -> str: 
#         """Helper function to format the Pydantic output from the extractor LLM."""
#         if not isinstance(output, extractor_output_schema):
#             print(f"Warning: Extractor LLM did not return expected Pydantic object. Output: {output}")
#             return "Could not extract the answer due to an internal formatting error."

#         if output.answer is None or "not found" in output.answer.lower():
#              return "I could not find a specific answer for that in the database."
#         source_str = ""
#         if output.source_code and output.source_clause:
#             source_str = f" (Source: {output.source_code}, Clause: {output.source_clause})"
#         elif output.source_code:
#              source_str = f" (Source: {output.source_code})"
#         return f"The answer is: **{output.answer}**{source_str}"

#     def route_extractor_logic(input_dict: dict):
#         """
#         NEW Step 3: Takes the single best document (or None) and routes
#         to the final extraction LLM.
#         """
#         top_doc_str = input_dict.get("top_doc", None)
#         print(f"--- Executing Extractor (Step 3: Final Extraction) ---")
        
#         if top_doc_str is None:
#             return RunnableLambda(lambda x: "I could not find a specific answer for that in the database.")
#         else:
           
#             def remap_for_extractor(inputs):
#                 return {
#                     "context": inputs.get("top_doc"),
#                     "query": inputs.get("query")
#                 }

#             extractor_prompt_obj = ChatPromptTemplate.from_template(extractor_prompt_template)
#             extractor_llm_chain = (
#                 RunnableLambda(remap_for_extractor)
#                 | extractor_prompt_obj 
#                 | llm.with_structured_output(extractor_output_schema, include_raw=False) 
#                 | format_extractor_output
#             )
#             return extractor_llm_chain


#     return (
 
#         RunnableLambda(hybrid_search_for_extractor)

#         | RunnableLambda(rerank_documents)

#         | RunnableLambda(route_extractor_logic)
#     )



# def create_supervisor_chain():
#     """Builds the master chain that routes tasks."""
#     try:
#         query_router_schema = prompts.QueryRouter
#         router_prompt_template = prompts.ROUTER_PROMPT
#     except AttributeError as e:
#         print(f"--- ERROR: Missing Pydantic model or prompt in prompts module: {e} ---")
#         sys.exit(1)

#     router_prompt_obj = ChatPromptTemplate.from_template(router_prompt_template)
#     router_chain = router_prompt_obj | llm.with_structured_output(query_router_schema)

#     print("Initializing chains...")
#     general_chain = create_general_query_chain()
#     extractor_chain = create_extractor_chain() 
#     print("All chains initialized.")

#     branch = RunnableBranch(
#         (lambda x: x.get('task_type') == 'extraction_query', extractor_chain),

#         general_chain 
#     )

#     return (

#         {"task": {"query": RunnablePassthrough()} | router_chain,
#          "query": RunnablePassthrough()} 
#         | RunnableLambda(lambda x: {
#             "task_type": getattr(x.get("task"), "task_type", "general_query"), 
#             "query": x.get("query"), # Pass the original query string
#             "thought": getattr(x.get("task"), "thought", "")
#           })
#         | RunnableLambda(lambda x: print(f"\n--- 🧠 Router Decision --- \nTask: {x['task_type']}\nThought: {x['thought']}\n{'-'*25}") or x) # Print router decision
#         | branch
#     )


# if __name__ == "__main__":
#     print("Road Safety Supervisor Agent Initialized.")
#     print("This file is for logic. Run 'streamlit run app.py' to use the app.")

#     if tools.sql_engine is None or tools.langchain_vector_store is None:
#         print("\n--- WARNING: One or more tools failed to initialize. ---")
#         print("--- Have you run 'python data_processor.py' first? ---")
#         sys.exit(1)

#     try:
#         agent_chain = create_supervisor_chain()
#     except Exception as e:
#         print(f"--- FATAL ERROR during agent chain creation: {e} ---")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)

#     # --- Automatic Test Run ---
#     print("\n--- INITIATING AUTOMATIC TEST RUN (FINAL VERSION) ---")
    

#     test_queries = ["What is the minimum height requirement for a road sign above a pedestrian area, and how far apart should two successive signs be on an urban road?",
#                     "I found a chevron sign that is 500 mm wide and 600 mm tall. Is this compliant with the standard dimensions for an 'Object Hazard (Left) Sign'?",
#                     "My school's 'SCHOOL KEEP CLEAR' road marking is almost invisible due to fading. What is the required colour and size, and what must I do to fix this compliance problem?"]

#     for i, query in enumerate(test_queries, 1):
#         print(f"\n{'='*80}")
#         print(f"--- 🧪 EXECUTING TEST QUERY {i}/{len(test_queries)} ---")
#         print(f"QUERY: {query}")
#         print(f"{'-'*80}\n")
        
#         try:
#             print("--- Agent Answer ---")
#             full_response = ""
#             # We must pass the query as a string, as the chain expects string input
#             for chunk in agent_chain.stream(query): 
#                 print(chunk, end="", flush=True)
#                 full_response += chunk
#             print("\n--------------------")

#         except KeyboardInterrupt: print("\nExiting..."); break
#         except Exception as e:
#              import traceback; print(f"\n--- Error During Query ---"); query_at_error = 'N/A'
#              try: query_at_error = query
#              except NameError: pass
#              print(f"Query: {query_at_error}"); print(f"Error: {e}"); traceback.print_exc(); print(f"-------------------------")
    
#     print(f"\n{'='*80}")
#     print("-- AUTOMATIC TEST RUN COMPLETE ---")
#     print("Exiting.")