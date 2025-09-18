import json
from typing import Any, Dict, Optional
from pocket_agent.agent import AgentEvent

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.rule import Rule
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class RichConsoleFormatter:
    """Rich-enhanced formatter for agent events with beautiful console output"""
    
    def __init__(self, console: Optional[Console] = None):
        if not RICH_AVAILABLE:
            raise ImportError(
                "Rich is not installed. Install it with: pip install rich\n"
                "Or use the regular ConsoleFormatter instead."
            )
        
        self.console = console or Console()
        
        # Define color schemes for different message types
        self.colors = {
            "user": "cyan",
            "user_sub": "bright_cyan", 
            "assistant": "green",
            "sub_agent": "yellow",
            "tool": "magenta",
            "system": "blue",
            "error": "red"
        }
        
        # Define emojis for different roles
        self.emojis = {
            "user": "👤",
            "user_sub": "📨", 
            "assistant": "🤖",
            "sub_agent": "🔧",
            "tool": "⚡",
            "system": "📡",
            "image": "🖼️",
            "attachment": "📎"
        }

    def format_event(self, event: AgentEvent) -> None:
        """Format and display an agent event using Rich components"""
        if not RICH_AVAILABLE:
            # Fallback to basic print if Rich is not available
            self._fallback_format(event)
            return
            
        if event.event_type == "new_message":
            self._format_message_event(event)
        else:
            self._format_other_event(event)

    def _format_message_event(self, event: AgentEvent) -> None:
        """Format a new_message event with Rich styling"""
        role = event.data["role"]
        name = event.meta.get("agent_name", None)
        is_sub_agent = event.meta.get("is_sub_agent", False)
        content = event.data.get("content", None)
        tool_calls = event.data.get("tool_calls", None)
        
        # For tool results, get the tool name from the message data
        tool_name = event.data.get("name") if role == "tool" else None
        tool_call_id = event.data.get("tool_call_id") if role == "tool" else None
        
        # Determine message type and styling
        message_type, color, emoji = self._get_message_style(role, is_sub_agent)
        
        # Create header text with enhanced tool result handling
        if role == "tool":
            # Special handling for tool results
            if tool_name and tool_call_id:
                short_id = tool_call_id[-6:] if len(tool_call_id) > 6 else tool_call_id
                header_text = f"{emoji} TOOL RESULT: {tool_name} (#{short_id})"
            elif tool_name:
                header_text = f"{emoji} TOOL RESULT: {tool_name}"
            else:
                header_text = f"{emoji} TOOL RESULT"
        elif name:
            if role == "user" and is_sub_agent:
                header_text = f"{emoji} AGENT MESSAGE → {name} (Sub-Agent)"
            elif role == "user":
                header_text = f"{emoji} USER MESSAGE → {name}"
            elif is_sub_agent:
                header_text = f"{emoji} {name} (Sub-Agent) MESSAGE"
            else:
                header_text = f"{emoji} {name} MESSAGE"
        else:
            if role == "user" and is_sub_agent:
                header_text = f"{emoji} AGENT MESSAGE → SUB-AGENT"
            elif role == "user":
                header_text = f"{emoji} USER MESSAGE"
            elif is_sub_agent:
                header_text = f"{emoji} SUB-AGENT MESSAGE"
            else:
                header_text = f"{emoji} {role.upper()} MESSAGE"
        
        # Create main panel content
        panel_content = []
        
        # Add content if present
        if content:
            content_text = self._format_content(content, role)
            if content_text:
                panel_content.append(content_text)
        
        # Add tool calls if present  
        if tool_calls:
            tool_table = self._create_tool_calls_table(tool_calls)
            panel_content.append(tool_table)
        
        # Create the main panel
        if panel_content:
            if len(panel_content) == 1:
                # Single item - pass it directly
                panel = Panel(
                    panel_content[0],
                    title=header_text,
                    title_align="left",
                    border_style=color,
                    box=box.ROUNDED,
                    padding=(1, 2)
                )
            else:
                # Multiple items - use Rich Group to combine them
                panel = Panel(
                    Group(*panel_content),
                    title=header_text,
                    title_align="left",
                    border_style=color,
                    box=box.ROUNDED,
                    padding=(1, 2)
                )
        else:
            panel = Panel(
                Text("(no content)", style="dim"),
                title=header_text,
                title_align="left", 
                border_style=color,
                box=box.ROUNDED,
                padding=(1, 2)
            )
        
        self.console.print()  # Empty line before
        self.console.print(panel)
        self.console.print()  # Empty line after

    def _get_message_style(self, role: str, is_sub_agent: bool) -> tuple[str, str, str]:
        """Get the appropriate styling for a message type"""
        if role == "user" and is_sub_agent:
            return ("user_sub", self.colors["user_sub"], self.emojis["user_sub"])
        elif role == "user":
            return ("user", self.colors["user"], self.emojis["user"])
        elif is_sub_agent:
            return ("sub_agent", self.colors["sub_agent"], self.emojis["sub_agent"])
        elif role == "assistant":
            return ("assistant", self.colors["assistant"], self.emojis["assistant"])
        elif role == "tool":
            return ("tool", self.colors["tool"], self.emojis["tool"])
        else:
            return ("system", self.colors["system"], self.emojis["system"])

    def _format_content(self, content, role: str = None) -> Optional[Text]:
        """Format message content with appropriate styling"""
        if isinstance(content, str):
            # Simple string content - try to detect if it's code/JSON
            if self._looks_like_json(content):
                try:
                    parsed = json.loads(content) 
                    formatted_json = json.dumps(parsed, indent=2)
                    return Syntax(formatted_json, "json", theme="github-dark", line_numbers=False)
                except json.JSONDecodeError:
                    pass
            elif self._looks_like_code(content):
                return Syntax(content, "python", theme="github-dark", line_numbers=False)
            else:
                # Regular text - check if it looks like markdown
                if self._looks_like_markdown(content):
                    return Markdown(content)
                else:
                    return Text(content)
        
        elif isinstance(content, list):
            # Multimodal content (common for tool results)
            parts = []
            for item in content:
                if item.get("type") == "text":
                    text = item.get('text', '')
                    if text:
                        # For tool results, try to format the text nicely
                        if role == "tool":
                            # Check if tool result text looks like structured data
                            if self._looks_like_json(text):
                                try:
                                    parsed = json.loads(text)
                                    formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                                    parts.append(Syntax(formatted, "json", theme="github-dark", line_numbers=False))
                                    continue
                                except json.JSONDecodeError:
                                    pass
                            # Add some nice formatting for tool results
                            parts.append(Text(text, style="bright_green"))
                        else:
                            parts.append(Text(text))
                elif item.get("type") == "image_url":
                    parts.append(Text(f"{self.emojis['image']} [Image attached]", style="dim cyan"))
                else:
                    content_type = item.get('type', 'unknown').title()
                    parts.append(Text(f"{self.emojis['attachment']} [{content_type} content]", style="dim"))
            
            if parts:
                combined = Text()
                for i, part in enumerate(parts):
                    if i > 0:
                        combined.append("\n\n")
                    combined.append(part)
                return combined
        
        return None

    def _create_tool_calls_table(self, tool_calls: list) -> Table:
        """Create a beautifully formatted table for tool calls"""
        table = Table(
            title=f"{self.emojis['tool']} Tool Calls ({len(tool_calls)})",
            title_style="bold magenta",
            box=box.ROUNDED,  # Changed from SIMPLE_HEAVY to ROUNDED for softer look
            show_header=True,
            border_style="magenta",
            title_justify="left",
            min_width=80,
            expand=True  # Make table fill available width
        )
        
        # Enhanced column design with better styling
        table.add_column("☟", style="bold bright_white", width=3, justify="center")
        table.add_column("🔧 Tool", style="bold bright_cyan", min_width=20, no_wrap=False)
        table.add_column("📋 Arguments", style="default", ratio=1)
        table.add_column("ID", style="bright_white", width=8, justify="right")  # New column for tool call ID
        
        for i, tool_call in enumerate(tool_calls, 1):
            function_info = tool_call.get('function', {})
            function_name = function_info.get('name', 'unknown')
            function_args = function_info.get('arguments', '{}')
            tool_call_id = tool_call.get('id', '')
            
            # Enhanced argument formatting
            try:
                args_dict = json.loads(function_args) if isinstance(function_args, str) else function_args
                
                if not args_dict or args_dict == {}:
                    # Better styling for empty arguments
                    args_display = Text("∅ no arguments", style="dim italic")
                    
                elif len(str(args_dict)) <= 50:  # Short arguments - display inline
                    formatted_args = json.dumps(args_dict, separators=(',', ': '))
                    args_display = Text(formatted_args, style="bright_green")
                    
                else:  # Long arguments - use syntax highlighting
                    formatted_args = json.dumps(args_dict, indent=2, separators=(',', ': '))
                    args_display = Syntax(
                        formatted_args, 
                        "json", 
                        theme="github-dark",  # Changed theme for better readability
                        line_numbers=False,
                        background_color="default",
                        word_wrap=True
                    )
                    
            except (json.JSONDecodeError, TypeError):
                # Better error handling and styling
                error_text = str(function_args)
                if len(error_text) > 100:
                    error_text = error_text[:97] + "..."
                args_display = Text(f"⚠️ {error_text}", style="bold red")
            
            # Enhanced function name styling
            if function_name == 'unknown':
                func_display = Text("❓ unknown", style="bold yellow")
            else:
                func_display = Text(function_name, style="bold bright_cyan")
            
            # Enhanced row styling
            row_style = None
            if i % 2 == 0:  # Alternate row styling
                row_style = "on grey11"
            
            # Truncate tool call ID for display
            display_id = tool_call_id[-6:] if tool_call_id else "──"
            
            table.add_row(
                str(i),
                func_display,
                args_display,
                display_id,
                style=row_style
            )
        
        return table

    def _format_other_event(self, event: AgentEvent) -> None:
        """Format non-message events"""
        title = f"{self.emojis['system']} Event: {event.event_type}"
        
        if event.data:
            try:
                formatted_data = json.dumps(event.data, indent=2)
                content = Syntax(formatted_data, "json", theme="monokai", line_numbers=False)
            except (TypeError, ValueError):
                content = Text(str(event.data))
        else:
            content = Text("(no data)", style="dim")
        
        panel = Panel(
            content,
            title=title,
            title_align="left",
            border_style=self.colors["system"],
            box=box.ROUNDED,
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(panel)
        self.console.print()

    def _looks_like_json(self, text: str) -> bool:
        """Check if text looks like JSON"""
        text = text.strip()
        return (text.startswith('{') and text.endswith('}')) or \
               (text.startswith('[') and text.endswith(']'))

    def _looks_like_code(self, text: str) -> bool:
        """Check if text looks like code"""
        code_indicators = ['def ', 'class ', 'import ', 'from ', 'if __name__', '```', 'function', 'const ', 'let ', 'var ']
        return any(indicator in text for indicator in code_indicators)

    def _looks_like_markdown(self, text: str) -> bool:
        """Check if text looks like markdown"""
        md_indicators = ['# ', '## ', '### ', '- ', '* ', '1. ', '**', '__', '[', '](', '`']
        return any(indicator in text for indicator in md_indicators)

    def _fallback_format(self, event: AgentEvent) -> None:
        """Fallback formatting when Rich is not available"""
        from .console_formatter import ConsoleFormatter
        fallback_formatter = ConsoleFormatter()
        fallback_formatter.format_event(event)


# Convenience function to create formatter with error handling
def create_rich_formatter(console: Optional[Console] = None) -> RichConsoleFormatter:
    """Create a RichConsoleFormatter with proper error handling"""
    try:
        return RichConsoleFormatter(console)
    except ImportError as e:
        raise ImportError(
            f"Cannot create RichConsoleFormatter: {e}\n"
            "Install Rich with: pip install rich"
        )
