"""
Command-line interface for the data analysis agent.
"""

import argparse
import os
import asyncio
from dotenv import load_dotenv
from agent.root import DataAgent

MODEL = "gpt-4.1"


async def run_single_query(agent: DataAgent, query: str):
    """Run a single query and display the result."""
    print(f"Question: {query}\n")

    final_answer = None
    async for event in agent.astream(query):
        event_type = event.get("type", "")
        content = event.get("content", "")
        metadata = event.get("metadata", {})
        is_subagent = metadata.get("is_subagent", False)
        subagent_name = metadata.get("subagent_name")

        # Show tool calls when AI decides to call a tool
        if "tool_calls" in event:
            for tool_call in event["tool_calls"]:
                tool_name = tool_call['name']
                if is_subagent:
                    # Indent subagent tool calls to show hierarchy
                    print(f"  ↳ [{subagent_name}] calling {tool_name}")
                else:
                    if tool_name == 'task':
                        print("→ calling stats sub-agent")
                    else:
                        print(f"→ calling {tool_name}")

        # Capture final answer (last AIMessage from main agent without tool calls)
        elif "AIMessage" in event_type and "tool_calls" not in event and not is_subagent:
            final_answer = content

    # Display final answer
    if final_answer:
        print(f"\nAnswer:\n{final_answer}")


async def run_interactive(agent: DataAgent):
    """Run the interactive mode."""
    print("=" * 60)
    print("SynMax Data Agent - Interactive Mode")
    print("=" * 60)
    print("Ask questions about your data in natural language.")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 60)
    print()

    while True:
        try:
            # Get user input (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            question = await loop.run_in_executor(None, lambda: input("\n> ").strip())

            # Check for exit commands
            if question.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break

            # Skip empty input
            if not question:
                continue

            # Process query with streaming
            print()
            final_answer = None

            async for event in agent.astream(question):
                event_type = event.get("type", "")
                content = event.get("content", "")
                metadata = event.get("metadata", {})
                is_subagent = metadata.get("is_subagent", False)
                subagent_name = metadata.get("subagent_name")

                # Show tool calls when AI decides to call a tool
                if "tool_calls" in event:
                    for tool_call in event["tool_calls"]:
                        tool_name = tool_call['name']
                        if is_subagent:
                            # Indent subagent tool calls to show hierarchy
                            print(f"  ↳ [{subagent_name}] calling {tool_name}")
                        else:
                            if tool_name == 'task':
                                print("→ calling stats sub-agent")
                            else:
                                print(f"→ calling {tool_name}")

                # Capture final answer (last AIMessage from main agent without tool calls)
                elif "AIMessage" in event_type and "tool_calls" not in event and not is_subagent:
                    final_answer = content

            # Display final answer
            if final_answer:
                print("\n" + "=" * 60)
                print("ANSWER:")
                print("=" * 60)
                print(final_answer)
                print("=" * 60)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


async def main_async(args):
    """Async main function."""
    # Initialize agent
    print("Initializing SynMax Data Agent...")
    agent = DataAgent(dataset_path=args.dataset_path, model=MODEL)
    print(f"Dataset path: {args.dataset_path}")
    print()

    # Single query mode
    if args.query:
        await run_single_query(agent, args.query)
    else:
        # Interactive mode
        await run_interactive(agent)


def main():
    """Run the CLI interface."""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="SynMax Data Agent - Natural language data analysis"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.getenv("DATASET_PATH", "./data/dataset.csv"),
        help="Path to the dataset file"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to run (non-interactive mode)"
    )

    args = parser.parse_args()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return

    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"Warning: Dataset not found at {args.dataset_path}")
        print("Please ensure the dataset is available or specify a different path with --dataset-path")
        print()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
