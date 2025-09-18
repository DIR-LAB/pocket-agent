from typing import Optional
from pocket_agent.agent import AgentEvent, AgentHooks
from rich_console_formatter import RichConsoleFormatter, RICH_AVAILABLE

try:
    from rich.console import Console
except ImportError:
    Console = None


class RichAgentHooks(AgentHooks):
    """Agent hooks that use Rich formatting for beautiful console output"""
    
    def __init__(self, console: Optional[Console] = None, fallback_to_basic: bool = True):
        """
        Initialize Rich-enabled hooks
        
        Args:
            console: Rich Console instance (creates one if None)
            fallback_to_basic: Whether to fallback to basic formatting if Rich unavailable
        """
        super().__init__()
        
        if RICH_AVAILABLE:
            self.formatter = RichConsoleFormatter(console)
        else:
            raise ImportError(
                "Rich is not available and fallback is disabled. "
                "Install Rich with: pip install rich"
            )
    
    async def on_event(self, event: AgentEvent) -> None:
        """Handle events with Rich formatting"""
        self.formatter.format_event(event)