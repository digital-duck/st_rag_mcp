import streamlit as st
import asyncio
import json
import logging
import os
import sqlite3
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Tuple
from fastmcp import Client
from datetime import datetime
import hashlib
import numpy as np

# RAG dependencies
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_util
    from sklearn.metrics.pairwise import cosine_similarity
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    st.warning("⚠️ RAG features disabled. Install: pip install sentence-transformers scikit-learn")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit page config
st.set_page_config(
    page_title="MCP Client with RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

SAMPLE_QUERIES = """
- 15 + 27
- what is 3 times 4
- 9 divide 3
- calculate power of 2 to 3
- compute square root of 2
- sine of 30 degrees
- tan(45 degree)
- find  arc tangent of 1 in degree
- get stock price of GOOG
- find company details for ticker GOOG
- Get company info about AAPL
- health check
- server diagnostics
- repeat this message: hello MCP server
- what is pi (fail)
"""

CUSTOM_CSS_STYLE = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .tool-call {
        background-color: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.375rem;
    }
    .error-message {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.375rem;
    }
    .debug-info {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.375rem;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .rag-match {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
    .similarity-score {
        background-color: #d4edda;
        color: #155724;
        padding: 0.2rem 0.4rem;
        border-radius: 0.2rem;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
"""

# Custom CSS
st.markdown(CUSTOM_CSS_STYLE, unsafe_allow_html=True)

# LLM Models and Provider Selection
LLM_PROVIDER_MAP = {
    "google": [
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ],
    "openai": [
        "gpt-4o-mini", 
        "gpt-4o", 
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022", 
        "claude-3-7-sonnet",
    ],
    # "ollama": [
    #     "qwen2.5",
    # ],
    # "bedrock": [
    #     "anthropic.claude-3-5-sonnet-20241022-v2:0",
    # ],
}

MCP_SERVER_PATH = "mcp_server.py"
SQLITE_DB_FILE = "mcp_chat_history.db"

TABLE_CHAT_HISTORY = "chat_history"
CHAT_HISTORY_DDL = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_CHAT_HISTORY} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        llm_provider TEXT,
        model_name TEXT,
        parsing_mode TEXT,
        user_query TEXT NOT NULL,
        parsed_action TEXT,
        tool_name TEXT,
        resource_uri TEXT,
        parameters TEXT,
        confidence REAL,
        reasoning TEXT,
        rag_matches TEXT,
        similarity_scores TEXT,
        response_data TEXT,
        formatted_response TEXT,
        elapsed_time_ms INTEGER,
        error_message TEXT,
        success BOOLEAN NOT NULL DEFAULT 1
    )
"""

def ask_llm(provider, client, model_name, query, system_prompt, max_tokens=300, temperature=0.1):
    # Get LLM response with dynamic prompt
    if provider == "anthropic":
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Query: {query}"}]
        )
        llm_response = response.content[0].text.strip()
    
    elif provider == "openai":
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        llm_response = response.choices[0].message.content.strip()
    
    elif provider == "google":
        response = client.generate_content(
            f"{system_prompt}\n\nUser Query: {query}",
            generation_config={
                "temperature": temperature, 
                "max_output_tokens": max_tokens,
            }
        )
        llm_response = response.text.strip()
    else:
        st.error(f"Unsupported LLM provider: {provider}")
        return None
    
    # Clean and parse JSON
    if llm_response.startswith("```json"):
        llm_response = llm_response.replace("```json", "").replace("```", "").strip()
    elif llm_response.startswith("```"):
        llm_response = llm_response.replace("```", "").strip()
    
    parsed_response = json.loads(llm_response)
    return parsed_response

# --- Database Operations ---
class ChatHistoryDB:
    def __init__(self, db_file: str = SQLITE_DB_FILE):
        self.db_file = db_file
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            
            # Create table with original schema first
            cursor.execute(CHAT_HISTORY_DDL)
            
            # # Check if new columns exist and add them if needed
            # cursor.execute("PRAGMA table_info(chat_history)")
            # columns = [column[1] for column in cursor.fetchall()]
            
            # if 'rag_matches' not in columns:
            #     cursor.execute("ALTER TABLE chat_history ADD COLUMN rag_matches TEXT")
            #     logging.info("✅ Added rag_matches column to database")
            
            # if 'similarity_scores' not in columns:
            #     cursor.execute("ALTER TABLE chat_history ADD COLUMN similarity_scores TEXT")
            #     logging.info("✅ Added similarity_scores column to database")
            
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_session_id ON {TABLE_CHAT_HISTORY}(session_id)")
            conn.commit()
    
    def insert_chat_entry(self, entry: Dict[str, Any]) -> int:
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {TABLE_CHAT_HISTORY} (
                    session_id, timestamp, llm_provider, model_name, parsing_mode,
                    user_query, parsed_action, tool_name, resource_uri, parameters,
                    confidence, reasoning, rag_matches, similarity_scores, response_data, 
                    formatted_response, elapsed_time_ms, error_message, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.get('session_id'), entry.get('timestamp'), entry.get('llm_provider'),
                entry.get('model_name'), entry.get('parsing_mode'), entry.get('user_query'),
                entry.get('parsed_action'), entry.get('tool_name'), entry.get('resource_uri'),
                entry.get('parameters'), entry.get('confidence'), entry.get('reasoning'),
                entry.get('rag_matches'), entry.get('similarity_scores'), entry.get('response_data'),
                entry.get('formatted_response'), entry.get('elapsed_time_ms'), 
                entry.get('error_message'), entry.get('success', True)
            ))
            entry_id = cursor.lastrowid
            conn.commit()
            return entry_id
    
    def get_chat_history(self, limit: int = 100, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            
            query = f"SELECT * FROM {TABLE_CHAT_HISTORY}"
            params = []
            
            if filters and filters.get('session_id'):
                query += " WHERE session_id = ?"
                params.append(filters['session_id'])
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

# --- RAG System for MCP Tools/Resources ---
class MCPRAGSystem:
    def __init__(self):
        self.model = None
        self.tool_embeddings = None
        self.resource_embeddings = None
        self.tool_contexts = []
        self.resource_contexts = []
        
        if RAG_AVAILABLE:
            self.initialize_model()
    
    def initialize_model(self):
        """Initialize sentence transformer model"""
        try:
            # Use a fast, efficient model for embeddings
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("✅ RAG System initialized with all-MiniLM-L6-v2")
        except Exception as e:
            logging.error(f"❌ Failed to initialize RAG model: {e}")
            self.model = None
    
    def build_rich_context(self, item: Dict, item_type: str) -> str:
        """Build rich context for tools and resources with examples and synonyms"""
        if item_type == "tool":
            name = item.get('name', '')
            desc = item.get('description', '')
            
            # Enhanced context with usage examples and synonyms
            context_map = {
                'calculator': """
Tool: calculator
Description: Performs mathematical arithmetic operations
Type: computation tool
Usage examples: 
- Basic math: "15 plus 27", "multiply 8 by 4", "divide 100 by 5"
- Advanced: "what's 2 to the power of 3", "square root calculation"
- Keywords: add, subtract, multiply, divide, power, math, compute, calculate
Synonyms: math, arithmetic, computation, calculate, compute
                """,
                'trig': """
Tool: trig  
Description: Trigonometric functions (sine, cosine, tangent)
Type: mathematical tool
Usage examples:
- "sine of 30 degrees", "cosine of 45", "tangent of 60 degrees"
- "sin(π/4)", "cos(0)", "tan(90 degrees)"
- Unit support: degrees, radians
Keywords: trigonometry, sine, cosine, tangent, sin, cos, tan, angle
Synonyms: trigonometry, trig functions, angles, geometry
                """,
                'health': """
Tool: health
Description: Server health check and status monitoring  
Type: diagnostic tool
Usage examples:
- "health check", "server status", "is server running"
- "ping server", "system status", "connectivity test"
Keywords: health, status, ping, check, monitor, diagnostic
Synonyms: status, ping, check, monitor, diagnostic, connectivity
                """,
                'echo': """
Tool: echo
Description: Echo back messages for testing
Type: utility tool  
Usage examples:
- "echo hello world", "repeat this message", "say hello"
- Testing connectivity and response
Keywords: echo, repeat, say, message, test
Synonyms: repeat, say, message, test, respond
                """
            }
            
            # Return specific context or generic one
            return context_map.get(name, f"""
Tool: {name}
Description: {desc}
Type: generic tool
Usage: General purpose tool for {name} operations
Keywords: {name}
            """).strip()
            
        elif item_type == "resource":
            # Handle both string URIs and AnyUrl objects safely
            uri_raw = item.get('uri', '')
            try:
                # Convert AnyUrl to string if needed
                uri = str(uri_raw) if hasattr(uri_raw, '__str__') else uri_raw
            except Exception:
                uri = 'unknown_resource'
            
            desc = item.get('description', '')
            
            # Build context for resources - use safe string operations
            try:
                uri_lower = uri.lower() if isinstance(uri, str) else str(uri).lower()
                
                if 'stock' in uri_lower:
                    return f"""
Resource: {uri}
Description: {desc}
Type: financial data resource
Usage examples:
- Stock information, financial data, company details
- Market data, stock prices, financial analysis  
Keywords: stock, finance, market, company, financial, investment
Synonyms: stocks, shares, equity, financial data, market data
                    """.strip()
                else:
                    # Safe extraction of resource name
                    try:
                        resource_name = uri.split('/')[-1] if '/' in str(uri) else str(uri)
                    except Exception:
                        resource_name = 'resource'
                    
                    return f"""
Resource: {uri}
Description: {desc}
Type: data resource
Usage: Access to {uri} data and information
Keywords: {resource_name}
                    """.strip()
            except Exception as e:
                # Fallback for any string operation failures
                logging.warning(f"Failed to process resource URI: {e}")
                return f"""
Resource: {uri}
Description: {desc}
Type: data resource
Usage: General data resource
Keywords: data, resource
                """.strip()
        
        return f"{item_type}: {item}"
    
    def build_embeddings(self, tools: List[Dict], resources: List[Dict]):
        """Build embeddings for all tools and resources"""
        if not self.model:
            return
        
        try:
            # Build rich contexts with error handling
            self.tool_contexts = []
            for tool in tools:
                try:
                    context = self.build_rich_context(tool, 'tool')
                    self.tool_contexts.append({
                        'name': tool.get('name', ''),
                        'description': tool.get('description', ''),
                        'context': context,
                        'type': 'tool'
                    })
                except Exception as e:
                    logging.warning(f"Failed to build context for tool {tool.get('name', 'unknown')}: {e}")
                    # Add fallback context
                    self.tool_contexts.append({
                        'name': tool.get('name', 'unknown'),
                        'description': tool.get('description', ''),
                        'context': f"Tool: {tool.get('name', 'unknown')}\nDescription: {tool.get('description', '')}",
                        'type': 'tool'
                    })
            
            self.resource_contexts = []
            for resource in resources:
                try:
                    context = self.build_rich_context(resource, 'resource')
                    # Safe URI extraction
                    uri_raw = resource.get('uri', '')
                    uri = str(uri_raw) if uri_raw else 'unknown_resource'
                    
                    self.resource_contexts.append({
                        'uri': uri,
                        'description': resource.get('description', ''),
                        'context': context,
                        'type': 'resource'
                    })
                except Exception as e:
                    logging.warning(f"Failed to build context for resource {resource.get('uri', 'unknown')}: {e}")
                    # Add fallback context
                    uri_raw = resource.get('uri', 'unknown_resource')
                    uri = str(uri_raw) if uri_raw else 'unknown_resource'
                    self.resource_contexts.append({
                        'uri': uri,
                        'description': resource.get('description', ''),
                        'context': f"Resource: {uri}\nDescription: {resource.get('description', '')}",
                        'type': 'resource'
                    })
            
            # Create embeddings with error handling
            if self.tool_contexts:
                try:
                    tool_texts = [item['context'] for item in self.tool_contexts]
                    self.tool_embeddings = self.model.encode(tool_texts)
                    logging.info(f"✅ Built embeddings for {len(self.tool_contexts)} tools")
                except Exception as e:
                    logging.error(f"❌ Failed to encode tool embeddings: {e}")
                    self.tool_embeddings = None
            
            if self.resource_contexts:
                try:
                    resource_texts = [item['context'] for item in self.resource_contexts]
                    self.resource_embeddings = self.model.encode(resource_texts)
                    logging.info(f"✅ Built embeddings for {len(self.resource_contexts)} resources")
                except Exception as e:
                    logging.error(f"❌ Failed to encode resource embeddings: {e}")
                    self.resource_embeddings = None
                
        except Exception as e:
            logging.error(f"❌ Failed to build embeddings: {e}")
            # Ensure we have empty but valid states
            self.tool_contexts = []
            self.resource_contexts = []
            self.tool_embeddings = None
            self.resource_embeddings = None
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform optimized semantic search across tools and resources using sentence-transformers util"""
        if not self.model:
            return []
        
        try:
            # Encode the query
            query_embedding = self.model.encode([query])
            
            # Combine tool and resource embeddings efficiently
            all_embeddings = []
            all_contexts = []
            
            if self.tool_embeddings is not None and len(self.tool_embeddings) > 0:
                all_embeddings.append(self.tool_embeddings)
                all_contexts.extend([(ctx, 'tool') for ctx in self.tool_contexts])
            
            if self.resource_embeddings is not None and len(self.resource_embeddings) > 0:
                all_embeddings.append(self.resource_embeddings)
                all_contexts.extend([(ctx, 'resource') for ctx in self.resource_contexts])
            
            if not all_embeddings:
                return []
            
            # Concatenate all embeddings into single corpus
            corpus_embeddings = np.concatenate(all_embeddings, axis=0)
            
            # Use optimized semantic_search from sentence-transformers
            search_results = st_util.semantic_search(
                query_embedding, 
                corpus_embeddings, 
                top_k=top_k,
                score_function=st_util.cos_sim
            )[0]  # Get results for first (and only) query
            
            # Map results back to our context format
            results = []
            for hit in search_results:
                corpus_id = hit['corpus_id']
                similarity = float(hit['score'])
                
                # Only include results above minimum threshold
                if similarity > 0.1:
                    context_item, item_type = all_contexts[corpus_id]
                    results.append({
                        'item': context_item,
                        'similarity': similarity,
                        'type': item_type
                    })
            
            logging.info(f"✅ Semantic search found {len(results)} relevant items for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logging.error(f"❌ Optimized semantic search failed: {e}")
            # Fallback to original method if needed
            return self._fallback_semantic_search(query, top_k)
    
    def _fallback_semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Fallback to original semantic search method if optimized version fails"""
        if not self.model:
            return []
        
        try:
            # Encode the query
            query_embedding = self.model.encode([query])
            results = []
            
            # Search tools using original method
            if self.tool_embeddings is not None and len(self.tool_embeddings) > 0:
                tool_similarities = cosine_similarity(query_embedding, self.tool_embeddings)[0]
                
                for i, similarity in enumerate(tool_similarities):
                    if similarity > 0.1:  # Minimum similarity threshold
                        results.append({
                            'item': self.tool_contexts[i],
                            'similarity': float(similarity),
                            'type': 'tool'
                        })
            
            # Search resources using original method
            if self.resource_embeddings is not None and len(self.resource_embeddings) > 0:
                resource_similarities = cosine_similarity(query_embedding, self.resource_embeddings)[0]
                
                for i, similarity in enumerate(resource_similarities):
                    if similarity > 0.1:  # Minimum similarity threshold
                        results.append({
                            'item': self.resource_contexts[i],
                            'similarity': float(similarity),
                            'type': 'resource'
                        })
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            logging.warning(f"⚠️ Used fallback search method for query: '{query[:50]}...'")
            return results[:top_k]
            
        except Exception as e:
            logging.error(f"❌ Fallback semantic search also failed: {e}")
            return []
    
    def build_dynamic_prompt(self, relevant_items: List[Dict], query: str) -> str:
        """Build dynamic system prompt based on relevant items"""
        if not relevant_items:
            return """
You are a tool selection assistant. 
Respond with ONLY a JSON object with action, tool, params, confidence, and reasoning fields.
"""
        
        # Build tools section
        tools_section = "Available tools:\n"
        for item in relevant_items:
            if item['type'] == 'tool':
                tool_info = item['item']
                similarity = item['similarity']
                tools_section += f"- {tool_info['name']}: {tool_info['description']} (relevance: {similarity:.2f})\n"
        
        # Build resources section
        resources_section = "\nAvailable resources:\n"
        for item in relevant_items:
            if item['type'] == 'resource':
                resource_info = item['item']
                similarity = item['similarity']
                resources_section += f"- {resource_info['uri']}: {resource_info['description']} (relevance: {similarity:.2f})\n"
        
        # Examples based on most relevant items
        examples_section = "\nExamples based on available tools:\n"
        
        # Add specific examples for discovered tools
        for item in relevant_items[:3]:  # Top 3 most relevant
            if item['type'] == 'tool':
                tool_name = item['item']['name']
                if tool_name == 'calculator':
                    examples_section += '- "15 plus 27" -> {"action": "tool", "tool": "calculator", "params": {"operation": "add", "num1": 15, "num2": 27}, "confidence": 0.98, "reasoning": "Simple addition"}\n'
                elif tool_name == 'trig':
                    examples_section += '- "sine of 30 degrees" -> {"action": "tool", "tool": "trig", "params": {"operation": "sine", "num1": 30, "unit": "degree"}, "confidence": 0.95, "reasoning": "Trigonometric calculation"}\n'
                elif tool_name == 'health':
                    examples_section += '- "health check" -> {"action": "tool", "tool": "health", "params": {}, "confidence": 0.9, "reasoning": "Server health check"}\n'
                elif tool_name == 'echo':
                    examples_section += '- "echo hello" -> {"action": "tool", "tool": "echo", "params": {"message": "hello"}, "confidence": 0.95, "reasoning": "Echo command"}\n'
        
        system_prompt = f"""
You are an intelligent tool selection assistant. 
Analyze the user query and respond with ONLY a JSON object:

{{
    "action": "tool",
    "tool": "tool_name_or_null",
    "params": {{"param1": "value1"}},
    "confidence": 0.95,
    "reasoning": "Brief explanation"
}}

{tools_section}

{resources_section}

{examples_section}

Instructions:
- Only use tools/resources listed above
- Consider the relevance scores when making decisions
- Set confidence based on query clarity and tool match
- If no tool matches well (all relevance < 0.3), set tool to null
- Respond with ONLY the JSON object, no other text.
"""

        return system_prompt

# Initialize session state
def init_session_state():
    if 'chat_history_db' not in st.session_state:
        st.session_state.chat_history_db = ChatHistoryDB()
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(f"{datetime.now()}{os.getpid()}".encode()).hexdigest()[:8]
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = "google"
    if 'use_llm' not in st.session_state:
        st.session_state.use_llm = True
    if 'server_connected' not in st.session_state:
        st.session_state.server_connected = False
    if 'available_tools' not in st.session_state:
        st.session_state.available_tools = []
    if 'available_resources' not in st.session_state:
        st.session_state.available_resources = []
    if 'use_rag' not in st.session_state:
        st.session_state.use_rag = True
    if 'last_parsed_query' not in st.session_state:
        st.session_state.last_parsed_query = None
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = MCPRAGSystem()
    if 'last_rag_matches' not in st.session_state:
        st.session_state.last_rag_matches = []

# --- Enhanced LLM Query Parser with RAG ---
class LLMQueryParser:
    def __init__(self, provider: str = "google"):
        self.provider = provider
        self.client = None
        self.model_name = None
        self.setup_llm_client()
    
    def setup_llm_client(self):
        try:
            if self.provider == "anthropic":
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.client = anthropic.Anthropic(api_key=api_key)
                    self.model_name = st.session_state.llm_model_name
            
            elif self.provider == "openai":
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                    self.model_name = st.session_state.llm_model_name
            
            elif self.provider == "google":
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.model_name = st.session_state.llm_model_name
                    self.client = genai.GenerativeModel(self.model_name)
                
        except Exception as e:
            st.error(f"Failed to initialize {self.provider}: {e}")
            self.client = None
    
    def parse_query_with_rag(self, query: str, rag_system: MCPRAGSystem) -> Optional[Dict[str, Any]]:
        """Parse query using RAG-enhanced semantic search"""
        if not self.client or not rag_system.model:
            return None
        
        try:
            # Perform semantic search
            relevant_items = rag_system.semantic_search(query, top_k=5)
            
            # Store RAG matches for debugging
            st.session_state.last_rag_matches = relevant_items
            
            if not relevant_items:
                # Fallback to standard parsing if no relevant items found
                return self.parse_query_sync(query)
            
            # Build dynamic prompt based on relevant items
            system_prompt = rag_system.build_dynamic_prompt(relevant_items, query)
            
            parsed_response = ask_llm(self.provider, self.client, self.model_name, query, system_prompt)
            if not parsed_response:
                return None

            # Add RAG metadata
            parsed_response['rag_enhanced'] = True
            parsed_response['rag_matches'] = len(relevant_items)
            
            if parsed_response.get("action") and parsed_response.get("confidence", 0) >= 0.3:
                return parsed_response
            
        except Exception as e:
            logging.error(f"RAG-enhanced parsing error: {e}")
            # Fallback to standard parsing
            return self.parse_query_sync(query)
        
        return None
    
    def parse_query_sync(self, query: str) -> Optional[Dict[str, Any]]:
        """Legacy parsing method with hardcoded examples"""
        if not self.client:
            return None
        
        system_prompt = """
You are a tool selection assistant. Respond with ONLY a JSON object:

{
    "action": "tool",
    "tool": "tool_name_or_null",
    "params": {"param1": "value1"},
    "confidence": 0.95,
    "reasoning": "Brief explanation"
}

Available tools:
- calculator: operation (add/subtract/multiply/divide/power), num1, num2
- trig: operation (sine/cosine/tangent), num1, unit (degree/radian)
- health: no parameters
- echo: message

Examples:
"15 plus 27" -> {"action": "tool", "tool": "calculator", "params": {"operation": "add", "num1": 15, "num2": 27}, "confidence": 0.98, "reasoning": "Simple addition"}
"sine of 30 degrees" -> {"action": "tool", "tool": "trig", "params": {"operation": "sine", "num1": 30, "unit": "degree"}, "confidence": 0.95, "reasoning": "Trigonometric calculation"}

Respond with ONLY the JSON object.
"""
        
        try:
            parsed_response = ask_llm(self.provider, self.client, self.model_name, query, system_prompt)
            if not parsed_response:
                return None

            parsed_response['rag_enhanced'] = False
            
            if parsed_response.get("action") and parsed_response.get("confidence", 0) >= 0.5:
                return parsed_response
            
        except Exception as e:
            st.error(f"LLM parsing error: {e}")
        
        return None

# --- Rule-based Parser ---
class RuleBasedQueryParser:
    @staticmethod
    def parse_query(query: str) -> Optional[Dict[str, Any]]:
        import re
        query_lower = query.lower().strip()
        
        # Health check
        if any(word in query_lower for word in ["health", "status", "ping"]):
            return {"action": "tool", "tool": "health", "params": {}, "confidence": 0.9, "reasoning": "Health check request", "rag_enhanced": False}
        
        # Echo command
        if query_lower.startswith("echo "):
            return {"action": "tool", "tool": "echo", "params": {"message": query[5:].strip()}, "confidence": 0.95, "reasoning": "Echo command", "rag_enhanced": False}
        
        # Calculator
        calc_patterns = [
            ("add", ["plus", "add", "+", "sum"]),
            ("subtract", ["minus", "subtract", "-"]),
            ("multiply", ["times", "multiply", "*", "×"]),
            ("divide", ["divide", "divided by", "/"]),
            ("power", ["power", "to the power", "^"])
        ]
        
        for operation, keywords in calc_patterns:
            for keyword in keywords:
                if keyword in query_lower:
                    numbers = re.findall(r'-?\d+(?:\.\d+)?', query)
                    if len(numbers) >= 2:
                        return {
                            "action": "tool",
                            "tool": "calculator", 
                            "params": {"operation": operation, "num1": float(numbers[0]), "num2": float(numbers[1])},
                            "confidence": 0.9,
                            "reasoning": f"Calculator operation: {operation}",
                            "rag_enhanced": False
                        }
        
        # Trig functions
        trig_patterns = [
            ("sine", ["sine", "sin"]),
            ("cosine", ["cosine", "cos"]),
            ("tangent", ["tangent", "tan"])
        ]
        
        for operation, keywords in trig_patterns:
            for keyword in keywords:
                if keyword in query_lower:
                    numbers = re.findall(r'-?\d+(?:\.\d+)?', query)
                    if numbers:
                        unit = "radian" if any(word in query_lower for word in ["radian", "rad"]) else "degree"
                        return {
                            "action": "tool",
                            "tool": "trig",
                            "params": {"operation": operation, "num1": float(numbers[0]), "unit": unit},
                            "confidence": 0.9,
                            "reasoning": f"Trigonometry: {operation}",
                            "rag_enhanced": False
                        }
        
        return None

# --- Utility Functions ---
def extract_result_data(result):
    try:
        if isinstance(result, list) and len(result) > 0:
            content_item = result[0]
            if hasattr(content_item, 'text'):
                try:
                    return json.loads(content_item.text)
                except json.JSONDecodeError:
                    return {"text": content_item.text}
            else:
                return {"content": str(content_item)}
        elif hasattr(result, 'content') and result.content:
            content_item = result.content[0]
            if hasattr(content_item, 'text'):
                try:
                    return json.loads(content_item.text)
                except json.JSONDecodeError:
                    return {"text": content_item.text}
            else:
                return {"content": str(content_item)}
        else:
            return result if isinstance(result, dict) else {"result": str(result)}
    except Exception as e:
        return {"error": f"Could not parse result: {e}"}

def format_result_for_display(tool_name: str, result: Dict) -> str:
    if isinstance(result, dict) and "error" in result:
        return f"❌ [Error] {result['error']}"
    
    if tool_name == "calculator":
        expression = result.get('expression', f"{result.get('num1', '?')} {result.get('operation', '?')} {result.get('num2', '?')} = {result.get('result', '?')}")
        return f"🧮 [Calculator] {expression}"
    
    elif tool_name == "trig":
        expression = result.get('expression', f"{result.get('operation', '?')}({result.get('num1', '?')}) = {result.get('result', '?')}")
        return f"📐 [Trigonometry] {expression}"
    
    elif tool_name == "health":
        return f"✅ [Health] {result.get('message', 'Server is healthy')}"
    
    elif tool_name == "echo":
        return f"🔊 [Echo] {result.get('echo', result.get('message', str(result)))}"
    
    return f"✅ [Result] {json.dumps(result, indent=2)}"

# --- CACHED MCP Operations using st.cache_resource ---
@st.cache_resource  
def get_mcp_server_info():
    """Get cached server info (tools/resources) - cached across reruns"""
    async def _discover():
        async with Client(MCP_SERVER_PATH) as client:
            # Get tools
            tools = await client.list_tools()
            available_tools = [{"name": tool.name, "description": tool.description} for tool in tools] if tools else []
            
            # Get resources
            try:
                resources = await client.list_resources()
                available_resources = [{"uri": resource.uri, "description": resource.description} for resource in resources] if resources else []
            except:
                available_resources = []
            
            return available_tools, available_resources
    
    return asyncio.run(_discover())

async def execute_mcp_query_async(parsed_query):
    """Execute MCP query with proper async context manager"""
    start_time = time.time()
    
    action = parsed_query.get("action")
    tool_name = parsed_query.get("tool")
    parameters = parsed_query.get("params", {})
    
    results = []
    
    if action == "tool" and tool_name:
        try:
            # Use proper async context manager for each query
            async with Client(MCP_SERVER_PATH) as client:
                tool_result = await client.call_tool(tool_name, parameters)
                tool_data = extract_result_data(tool_result)
                results.append({
                    "type": "tool",
                    "name": tool_name,
                    "data": tool_data,
                    "success": "error" not in tool_data
                })
        except Exception as e:
            results.append({
                "type": "error",
                "message": f"Tool call error: {e}",
                "success": False
            })
    
    elapsed_time = int((time.time() - start_time) * 1000)
    return results, elapsed_time


def do_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info(f"📍 [Session ID] `{st.session_state.session_id}`")
        
        # RAG System Status
        st.subheader("🧠 RAG System")
        if RAG_AVAILABLE and st.session_state.rag_system.model:
            st.success("✅ RAG System Active")
            st.info("🔍 Semantic search enabled")
            
            # RAG settings
            st.session_state.use_rag = st.checkbox(
                "🎯 Use RAG-Enhanced Parsing",
                value=st.session_state.use_rag,
                help="Use semantic search to find relevant tools dynamically"
            )
        else:
            st.error("❌ RAG System Disabled")
            st.warning("Install: `pip install sentence-transformers scikit-learn`")
            st.session_state.use_rag = False
        
        # LLM Provider/Model Selection
        c1, c2 = st.columns([2,3])
        with c1:
            LLM_PROVIDER_LIST = list(LLM_PROVIDER_MAP.keys())
            st.session_state.llm_provider = st.selectbox(
                "🤖 LLM Provider",
                LLM_PROVIDER_LIST,
                index=LLM_PROVIDER_LIST.index("google")
            )

        with c2:
            LLM_MODEL_NAME_LIST = LLM_PROVIDER_MAP.get(st.session_state.llm_provider)
            st.session_state.llm_model_name = st.selectbox(
                "Model Name",
                LLM_MODEL_NAME_LIST,
                index=0
            )

        # Parsing Mode
        st.session_state.use_llm = st.checkbox(
            "🧠 Use LLM Parsing",
            value=st.session_state.use_llm,
            key="cfg_use_llm",
        )
        
        # Parsing Mode Display
        if st.session_state.use_llm and st.session_state.use_rag and RAG_AVAILABLE:
            st.info("🎯 [Mode] RAG-enhanced LLM")
        elif st.session_state.use_llm:
            st.info("🤖 [Mode] LLM-based")
        else:
            st.info("📝 [Mode] Rule-based")
        
        # API Keys Status
        st.subheader("🔑 API Keys Status")
        api_keys_status = {
            "OpenAI": "✅" if os.getenv("OPENAI_API_KEY") else "❌",
            "Anthropic": "✅" if os.getenv("ANTHROPIC_API_KEY") else "❌",
            "Google": "✅" if os.getenv("GEMINI_API_KEY") else "❌",
        }
        
        for provider, status in api_keys_status.items():
            st.write(f"{status} {provider}")
        
        # Server Connection using st.cache_resource for discovery only!
        st.subheader("🔌 Server Status")
        
        # Try to get cached server info (tools/resources)
        try:
            # This will use cached discovery if available
            tools, resources = get_mcp_server_info()
            
            # If we get here, server is reachable
            st.session_state.server_connected = True
            st.session_state.available_tools = tools
            st.session_state.available_resources = resources
            
            # Build RAG embeddings when tools/resources are available
            if RAG_AVAILABLE and st.session_state.rag_system.model:
                st.session_state.rag_system.build_embeddings(tools, resources)
            
        except Exception as e:
            st.session_state.server_connected = False
            st.session_state.available_tools = []
            st.session_state.available_resources = []
        
        if st.button("🔄 Refresh Server Discovery"):
            # Clear cache and rediscover
            st.cache_resource.clear()
            st.rerun()
        
        # Connection Status
        if st.session_state.server_connected:
            st.success("🟢 Server Connected")
            
            # Show tools and resources if connected
            if st.session_state.available_tools:
                with st.expander("🔧 Available Tools"):
                    for tool in st.session_state.available_tools:
                        st.write(f"• [{tool['name']}] {tool['description']}")
            
            if st.session_state.available_resources:
                with st.expander("📚 Available Resources"):
                    for resource in st.session_state.available_resources:
                        st.write(f"• [{resource['uri']}] {resource['description']}")
            
            # RAG Embeddings Status
            if RAG_AVAILABLE and st.session_state.rag_system.model:
                tool_count = len(st.session_state.rag_system.tool_contexts)
                resource_count = len(st.session_state.rag_system.resource_contexts)
                st.info(f"🎯 RAG: {tool_count} tools, {resource_count} resources indexed. \nDynamic resource is unavailable via list_resources()")
        else:
            st.error("🔴 Server Disconnected")
            st.info(f"💡 Make sure {MCP_SERVER_PATH} is running, then click 'Refresh Server Discovery'")
        
        # Example queries
        st.subheader("💡 Example Queries")
        st.markdown(SAMPLE_QUERIES)
        # if st.button("15 + 27"):
        #     st.session_state.example_query = "15 + 27"

# --- Main App ---
def main():
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 MCP Client with RAG</h1>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    do_sidebar()

    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Query Interface")
        
        # Query Input
        default_query = st.session_state.get('example_query', '')
        user_query = st.text_input(
            "🎯 Enter your query:",
            value=default_query,
            placeholder="compute the square root of 144"
        )
        
        # Clear example after using it
        if 'example_query' in st.session_state:
            del st.session_state.example_query
        
        col_submit, col_clear = st.columns([1, 1])
        with col_submit:
            submit_button = st.button("🚀 Submit Query", type="primary")
        with col_clear:
            if st.button("🗑️ Clear Session"):
                st.session_state.session_id = hashlib.md5(f"{datetime.now()}{os.getpid()}".encode()).hexdigest()[:8]
                st.session_state.last_parsed_query = None
                st.session_state.last_rag_matches = []
                st.success("✅ New session started!")
                st.rerun()
        
        # Process Query using RAG!
        if submit_button and user_query:
            try:
                # Parse query
                parsed_query = None
                model_name = None
                rag_matches = []
                
                if st.session_state.use_llm:
                    parser = LLMQueryParser(st.session_state.llm_provider)
                    if parser.client:
                        # Use RAG-enhanced parsing if available
                        if st.session_state.use_rag and RAG_AVAILABLE and st.session_state.rag_system.model:
                            parsed_query = parser.parse_query_with_rag(user_query, st.session_state.rag_system)
                            rag_matches = st.session_state.last_rag_matches
                        else:
                            parsed_query = parser.parse_query_sync(user_query)
                        model_name = parser.model_name
                    else:
                        st.warning("🔄 LLM not available, using rule-based parsing")
                        parsed_query = RuleBasedQueryParser.parse_query(user_query)
                else:
                    parsed_query = RuleBasedQueryParser.parse_query(user_query)
                
                if parsed_query:
                    # Store for debug display
                    st.session_state.last_parsed_query = parsed_query
                    
                    # Execute query with proper async context manager
                    results, elapsed_time = asyncio.run(execute_mcp_query_async(parsed_query))
                    st.session_state.elapsed_time = elapsed_time

                    # Auto-update connection status if successful
                    if results and any(r.get('success', True) for r in results):
                        st.session_state.server_connected = True
                    
                    # Save to database with RAG data
                    db_entry = {
                        'session_id': st.session_state.session_id,
                        'timestamp': datetime.now(),
                        'llm_provider': st.session_state.llm_provider if st.session_state.use_llm else None,
                        'model_name': model_name,
                        'parsing_mode': 'RAG-Enhanced' if parsed_query.get('rag_enhanced') else ('LLM' if st.session_state.use_llm else 'Rule-based'),
                        'user_query': user_query,
                        'parsed_action': parsed_query.get('action'),
                        'tool_name': parsed_query.get('tool'),
                        'parameters': json.dumps(parsed_query.get('params', {})),
                        'confidence': parsed_query.get('confidence'),
                        'reasoning': parsed_query.get('reasoning'),
                        'rag_matches': json.dumps([{
                            'name': m['item'].get('name', m['item'].get('uri', '')),
                            'similarity': m['similarity'],
                            'type': m['type']
                        } for m in rag_matches]) if rag_matches else None,
                        'similarity_scores': json.dumps([m['similarity'] for m in rag_matches]) if rag_matches else None,
                        'elapsed_time_ms': elapsed_time,
                        'success': all(result.get('success', True) for result in results)
                    }
                    
                    entry_id = st.session_state.chat_history_db.insert_chat_entry(db_entry)
                    st.session_state.entry_id = entry_id

                    # Show RAG matches if available
                    if rag_matches:
                        st.markdown("### 🎯 RAG Search Results:")
                        for i, match in enumerate(rag_matches[:3]):  # Show top 3
                            item = match['item']
                            similarity = match['similarity']
                            item_type = match['type']
                            
                            if item_type == 'tool':
                                name = item.get('name', 'Unknown')
                                desc = item.get('description', '')
                            else:
                                name = item.get('uri', 'Unknown')
                                desc = item.get('description', '')
                            
                            st.markdown(f"""
                            <div class="rag-match">
                                <strong>#{i+1} {item_type.title()}:</strong> {name}<br>
                                <small>{desc}</small><br>
                                <span class="similarity-score">Similarity: {similarity:.3f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    for result in results:
                        if result['type'] == 'tool':
                            formatted_display = format_result_for_display(result['name'], result['data'])
                            st.markdown(f'<div class="tool-call">{formatted_display}</div>', unsafe_allow_html=True)
                        elif result['type'] == 'error':
                            st.markdown(f'<div class="error-message">❌ {result["message"]}</div>', unsafe_allow_html=True)

                    

                else:
                    st.error("❓ Failed to call MCP tool because I couldn't understand your query.")


                    
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                st.info("💡 Try clicking 'Refresh Server Discovery' if connection issues persist")
    
    with col2:
        st.subheader("📊 Query Analysis")


        # Display debug info
        if st.session_state.last_parsed_query:
            parsed_query = st.session_state.last_parsed_query
            parsing_mode = "RAG-Enhanced" if parsed_query.get('rag_enhanced') else "Legacy"
            st.success(f"✅ Query processed in {st.session_state.elapsed_time}ms using {parsing_mode} parsing (Entry ID: {st.session_state.entry_id})")
            
            st.markdown('<div class="debug-info">', unsafe_allow_html=True)
            st.markdown("🔍 Debug - Parsed Query:")
            debug_info = {
                "Action": parsed_query.get('action'),
                "Tool": parsed_query.get('tool'),
                "Parameters": parsed_query.get('params', {}),
                "Confidence": parsed_query.get('confidence'),
                "Reasoning": parsed_query.get('reasoning'),
                "RAG Enhanced": parsed_query.get('rag_enhanced', False),
                "RAG Matches": parsed_query.get('rag_matches', 0)
            }
            st.json(debug_info)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # RAG matches details
        if st.session_state.last_rag_matches:
            with st.expander("🎯 Detailed RAG Matches"):
                for i, match in enumerate(st.session_state.last_rag_matches):
                    item = match['item']
                    similarity = match['similarity']
                    item_type = match['type']
                    
                    st.write(f"[Match #{i+1} ({item_type})]")
                    st.write(f"• Similarity: {similarity:.4f}")
                    if item_type == 'tool':
                        st.write(f"• Tool: {item.get('name', 'Unknown')}")
                        st.write(f"• Description: {item.get('description', 'No description')}")
                    else:
                        st.write(f"• Resource: {item.get('uri', 'Unknown')}")
                        st.write(f"• Description: {item.get('description', 'No description')}")
                    st.write("---")
        
        # Session stats
        try:
            recent_entries = st.session_state.chat_history_db.get_chat_history(
                limit=5, 
                filters={'session_id': st.session_state.session_id}
            )
            
            if recent_entries:

                with st.expander("[Session Statistics]"):

                    latest_entry = recent_entries[0]
                    st.info(f"🔍 [Parser] {latest_entry['parsing_mode']}")
                    if latest_entry['model_name']:
                        st.info(f"🤖 [Model] {latest_entry['model_name']}")
                    
                    if len(recent_entries) > 1:
                        successful = sum(1 for entry in recent_entries if entry['success'])
                        avg_time = sum(entry['elapsed_time_ms'] or 0 for entry in recent_entries) / len(recent_entries)
                        rag_enhanced_count = sum(1 for entry in recent_entries if entry['parsing_mode'] == 'RAG-Enhanced')
                        
                        st.metric("Queries", len(recent_entries))
                        st.metric("Success Rate", f"{(successful/len(recent_entries)*100):.1f}%")
                        st.metric("Avg Response Time", f"{avg_time:.0f}ms")
                        st.metric("RAG-Enhanced", f"{rag_enhanced_count}/{len(recent_entries)}")
            else:
                st.info("💡 No queries in this session yet. Try asking something!")
                
        except Exception as e:
            st.error(f"Error loading query analysis: {e}")
        

if __name__ == "__main__":
    main()