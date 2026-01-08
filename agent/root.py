import os
from typing import Dict, Any
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from agent.react_agent import create_react_agent
from agent.tools import execute_python_subprocess, get_dataset_schema_and_sample
from agent.prompts import (
    DATA_SCIENCE_AGENT_SYSTEM_PROMPT,
    STATISTICS_SUBAGENT_SYSTEM_PROMPT,
)

# Try to import Langfuse (optional dependency)
try:
    from langfuse.langchain import CallbackHandler

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    CallbackHandler = None


class DataAgent:
    """
    Plan and Execute agent for data analysis with streaming capabilities.

    Architecture:
    - Main coordinator agent: Routes queries and handles simple retrieval
    - Statistics subagent: Handles advanced statistical analysis with proper methodology
    - Python subprocess execution: Secure code execution with safety checks (blocks writes, allows reads)

    The coordinator delegates complex tasks (correlations, patterns, forecasting, etc.)
    to the specialized statistics subagent which has rigorous training in:
    - Proper categorical variable encoding (binary vs nominal vs ordinal)
    - Appropriate statistical tests for different variable type combinations
    - Effect size reporting and confidence intervals
    """

    def __init__(
        self,
        dataset_path: str = None,
        model: str = "gpt-4.1",
        temperature: float = 0.1,
        enable_langfuse: bool = True,
    ):
        """
        Initialize the DataAgent.

        Args:
            dataset_path: Path to the dataset file. Falls back to DATASET_PATH env var.
            model: OpenAI model to use (default: gpt-4)
            temperature: Temperature for LLM responses (default: 0.1 for consistency)
            enable_langfuse: Enable Langfuse logging (default: True, requires env vars)
        """
        self.dataset_path = dataset_path or os.environ.get(
            "DATASET_PATH", "./dataset.csv"
        )
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")

        # Initialize Langfuse callback handler (optional)
        self.langfuse_handler = None
        if enable_langfuse and LANGFUSE_AVAILABLE:
            try:
                # Check if Langfuse env vars are set
                public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
                secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

                if public_key and secret_key:
                    self.langfuse_handler = CallbackHandler()
                    print("✓ Langfuse logging enabled")
                else:
                    print("⚠ Langfuse env vars not set, logging disabled")
            except Exception as e:
                print(f"⚠ Failed to initialize Langfuse: {e}")
        elif enable_langfuse and not LANGFUSE_AVAILABLE:
            print("⚠ Langfuse not installed. Install with: pip install langfuse")

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model, temperature=temperature, api_key=self.api_key
        )

        # Load dataset schema and sample rows
        self.dataset_context = get_dataset_schema_and_sample(self.dataset_path)

        # Create memory saver for conversation history
        self.memory = InMemorySaver()

        # Format system prompt with dataset context and path
        system_prompt = DATA_SCIENCE_AGENT_SYSTEM_PROMPT.format(
            dataset_context=self.dataset_context,
            dataset_path=self.dataset_path
        )

        # Format statistics subagent prompt with dataset context and path
        stats_system_prompt = STATISTICS_SUBAGENT_SYSTEM_PROMPT.format(
            dataset_context=self.dataset_context,
            dataset_path=self.dataset_path
        )

        # Create statistics subagent configuration
        statistics_subagent = {
            "name": "stats-agent",
            "description": "Expert statistician for advanced analysis including correlations, patterns, anomalies, forecasting, and statistical tests. Handles all categorical variable encoding properly.",
            "system_prompt": stats_system_prompt,
            "tools": [execute_python_subprocess],
        }

        # Create the deep agent with system prompt, tools, and subagents
        self.agent = create_react_agent(
            model=self.llm,
            tools=[execute_python_subprocess],
            system_prompt=system_prompt,
            checkpointer=self.memory,
            subagents=[statistics_subagent],
        )

    async def astream(self, question: str, thread_id: str = "default"):
        """
        Stream agent events in real-time as the agent processes the query (async).

        Args:
            question: The user's natural language question
            thread_id: Thread ID for conversation continuity (default: "default")

        Yields:
            Dict containing event information with keys:
                - type: Event type (e.g., "message", "tool_call", "thinking")
                - content: Event content
                - metadata: Additional event metadata

        Example:
            >>> agent = DataAgent(dataset_path="data.csv")
            >>> async for event in agent.astream("How many records are there?"):
            >>>     print(f"{event['type']}: {event['content']}")
        """
        # Build config with thread_id and optional Langfuse callback
        config = {"configurable": {"thread_id": thread_id}}
        if self.langfuse_handler:
            config["callbacks"] = [self.langfuse_handler]

        # Create input message
        input_message = {"messages": [HumanMessage(content=question)]}

        # Stream events from the agent with subgraph support
        # Format: (namespace_tuple, state_dict)
        # - Parent: ((), {'messages': [...]})
        # - Subgraph: (('subgraph_node:uuid',), {'messages': [...]})
        async for event in self.agent.astream(
            input_message, config, stream_mode="values", subgraphs=True
        ):
            # Parse the tuple: (namespace, state)
            namespace, state = event

            # Determine if this is from a subgraph
            is_subagent = len(namespace) > 0
            subagent_name = namespace[0].split(':')[0] if is_subagent and namespace else None

            # Extract the last message from the state
            if "messages" in state and len(state["messages"]) > 0:
                last_message = state["messages"][-1]

                event_data = {
                    "type": last_message.__class__.__name__,
                    "content": last_message.content
                    if hasattr(last_message, "content")
                    else str(last_message),
                    "metadata": {
                        "thread_id": thread_id,
                        "message_count": len(state["messages"]),
                        "is_subagent": is_subagent,
                        "subagent_name": subagent_name,
                    },
                }

                # Add tool call information if present
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    event_data["tool_calls"] = [
                        {"name": tc.get("name", "unknown"), "id": tc.get("id", "")}
                        for tc in last_message.tool_calls
                    ]

                # Add tool name if this is a tool message
                if hasattr(last_message, "name"):
                    event_data["tool_name"] = last_message.name

                yield event_data

    async def aquery(self, question: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Execute a complete query and return the final results (async, non-streaming).

        Args:
            question: The user's natural language question
            thread_id: Thread ID for conversation continuity (default: "default")

        Returns:
            Dict containing:
                - question: The original question
                - answer: The final synthesized answer
                - conversation: Full conversation history
                - metadata: Additional information about the query

        Example:
            >>> agent = DataAgent(dataset_path="data.csv")
            >>> result = await agent.aquery("What's the average revenue by category?")
            >>> print(result["answer"])
        """
        # Build config with thread_id and optional Langfuse callback
        config = {"configurable": {"thread_id": thread_id}}
        if self.langfuse_handler:
            config["callbacks"] = [self.langfuse_handler]

        # Create input message
        input_message = {"messages": [HumanMessage(content=question)]}

        # Invoke the agent and get final state
        final_state = await self.agent.ainvoke(input_message, config)

        # Extract messages from final state
        messages = final_state.get("messages", [])

        # Get the last AI message as the answer
        answer = messages[-1].content

        return {
            "question": question,
            "answer": answer,
            "conversation": [
                {
                    "role": msg.__class__.__name__,
                    "content": msg.content if hasattr(msg, "content") else str(msg),
                }
                for msg in messages
            ],
            "metadata": {
                "thread_id": thread_id,
                "message_count": len(messages),
                "dataset_path": self.dataset_path,
            },
        }
