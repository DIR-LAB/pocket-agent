import streamlit as st
import asyncio
import json
import os
from pathlib import Path
import litellm
from agent import SimpleAgent
from pocket_agent import AgentConfig, AgentEvent, AgentHooks
import threading
import time



class StreamlitAgentHooks(AgentHooks):
    """Custom hooks for Streamlit real-time updates"""
    
    def __init__(self, real_time_placeholder):
        super().__init__()
        self.real_time_placeholder = real_time_placeholder
    
    async def on_event(self, event: AgentEvent) -> None:
        """Handle agent events for real-time UI updates"""
        if event.event_type == "new_message":
            # Initialize agent_messages if not exists
            if "agent_messages" not in st.session_state:
                st.session_state.agent_messages = []
            
            message = event.data
            role = message.get("role", "unknown")
            
            # Format the message for display
            if role == "assistant":
                content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])
                
                if tool_calls:
                    # Show tool calls being made
                    tool_names = [tc.get('function', {}).get('name', 'unknown') for tc in tool_calls]
                    st.session_state.agent_messages.append({
                        "role": "assistant", 
                        "content": f"🔧 Using tools: {', '.join(tool_names)}",
                        "type": "tool_call"
                    })
                    # Update UI immediately
                    self.update_real_time_display()
                
                if content:
                    # Don't add assistant text message here - we'll handle it at the end
                    pass
            
            elif role == "tool":
                # Show tool results
                tool_name = message.get("name", "unknown")
                content = message.get("content", [])
                
                # Extract text content from tool result
                text_content = ""
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content += item.get("text", "")
                elif isinstance(content, str):
                    text_content = content
                
                # Truncate long tool results for display
                display_content = text_content[:150] + "..." if len(text_content) > 150 else text_content
                
                st.session_state.agent_messages.append({
                    "role": "tool",
                    "content": f"📊 **{tool_name}**: {display_content}",
                    "type": "tool_result"
                })
                # Update UI immediately
                self.update_real_time_display()
    
    def update_real_time_display(self):
        """Update the real-time display placeholder"""
        if "agent_messages" in st.session_state and st.session_state.agent_messages:
            with self.real_time_placeholder.container():
                st.markdown("#### 🔍 Real-time Agent Activity")
                for msg in st.session_state.agent_messages:
                    if msg["type"] == "tool_call":
                        st.info(msg["content"])
                    elif msg["type"] == "tool_result":
                        st.success(msg["content"])



def initialize_agent(real_time_placeholder):
    """Initialize the agent with proper configuration"""
    try:
        mcp_config = {
            "mcpServers": {
                "weather": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["server.py"],
                    "cwd": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "servers", "simple_weather")
                }
            }
        }
        
        # Get existing agent messages from session state
        existing_messages = st.session_state.get("agent_conversation_history", [])
        
        # Agent configuration with existing messages
        config = AgentConfig(
            llm_model="gpt-5-nano",
            system_prompt="You are a helpful assistant. Use the available tools to provide accurate responses. Be friendly and conversational.",
            messages=existing_messages  # Pass existing conversation history
        )
        
        # Create custom hooks for Streamlit
        hooks = StreamlitAgentHooks(real_time_placeholder)
        
        # Create agent with the custom hooks
        agent = SimpleAgent(
            agent_config=config,
            mcp_config=mcp_config,
            hooks=hooks
        )
        
        return agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return None

async def get_tools_from_agent(agent):
    """Get available tools from the agent"""
    try:
        tools = await agent.mcp_client.get_tools(format="mcp")
        
        # Format tools for display
        tools_info = []
        for tool in tools:
            tool_info = {
                "name": tool.name,
                "description": tool.description or "No description available"
            }
            tools_info.append(tool_info)
            
        return tools_info
    except Exception as e:
        st.error(f"Failed to fetch available tools: {str(e)}")
        return []

def run_async_agent_response(agent, user_input):
    """Run the agent response in a proper async context"""
    try:
        # Clear previous agent messages for new request
        st.session_state.agent_messages = []
        
        # Use asyncio.run() which properly handles event loop lifecycle
        response = asyncio.run(agent.run(user_input))
        
        # Save agent's conversation history to session state after each run
        st.session_state.agent_conversation_history = agent.messages.copy()
        
        return response
    except Exception as e:
        st.error(f"Error getting agent response: {str(e)}")
        return {"message": "Sorry, I encountered an error processing your request."}

def main():
    st.set_page_config(
        page_title="Chat Assistant",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 Chat Assistant")
    st.markdown("Ask me about anything! Watch as I think and use tools in real-time.")
    
    # Initialize agent conversation history if not exists
    if "agent_conversation_history" not in st.session_state:
        st.session_state.agent_conversation_history = []

    # Initialize agent first so we can use it in the sidebar
    real_time_placeholder = st.empty()  # Temporary placeholder
    agent = initialize_agent(real_time_placeholder)
    
    # Sidebar with real-time activity and information
    with st.sidebar:
        
        # Create a placeholder for real-time updates in the sidebar
        real_time_placeholder = st.empty()
        
        st.markdown("---")  # Separator line
        
        # Initialize agent with the placeholder - agent will now get existing messages
        if agent is None:
            st.error("Failed to initialize the agent. Please check the configuration.")
            st.stop()
        
        # Dynamic tools section - get tools from the initialized agent
        st.markdown("### Available Tools:")
        try:
            tools_info = asyncio.run(get_tools_from_agent(agent))
            
            if tools_info:
                tools_markdown = ""
                for tool in tools_info:
                    tools_markdown += f"- **{tool['name']}**: {tool['description']}\n"
                st.markdown(tools_markdown)
            else:
                st.markdown("*No tools available or unable to load tools*")
        except Exception as e:
            st.markdown(f"*Error loading tools: {str(e)}*")
        
        st.markdown("---")
        
        st.markdown("### Tips:")
        st.markdown("""
        - Ask specific questions to get better results
        - The assistant can use multiple tools to help you
        - Watch the real-time activity above to see what's happening
        """)
        
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.agent_messages = []
            st.session_state.agent_conversation_history = []
            real_time_placeholder.empty()
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Hello! I'm your AI assistant. Ask me anything and I'll use the available tools to help you!"
            })
            st.rerun()
        
    # Re-initialize agent with the correct placeholder - agent will now get existing messages
    agent = initialize_agent(real_time_placeholder)
    
    # Initialize session state for conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I'm your AI assistant. Ask me anything and I'll use the available tools to help you!"
        })
    
    # Initialize agent messages for real-time display
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to conversation
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response with real-time updates
        with st.chat_message("assistant"):
            with st.spinner("Thinking and using tools..."):
                try:
                    # The event handler will update the real_time_placeholder during execution
                    response = run_async_agent_response(agent, prompt)
                    
                    # Extract the message content from the response
                    if isinstance(response, dict):
                        if "message" in response:
                            content = response["message"]
                        elif "message_content" in response:
                            content = response["message_content"]
                        else:
                            content = str(response)
                    else:
                        content = str(response)
                    
                    # Add assistant response to conversation
                    st.session_state.messages.append({"role": "assistant", "content": content})
                    
                    # Clear the real-time display
                    real_time_placeholder.empty()
                    
                    # Show final response
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    # Clear real-time display on error
                    real_time_placeholder.empty()

if __name__ == "__main__":
    main()
