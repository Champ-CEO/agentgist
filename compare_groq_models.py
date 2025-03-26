"""
Compare Groq models to determine which is better for general tasks in AgentGist.

This script tests llama3-groq-70b-tool-use and llama-3.3-70b-versatile on:
1. General question answering
2. Tool use capabilities
3. Structured output formatting
4. Reddit post analysis (specific to AgentGist)
"""

import os
import json
import time
from typing import Any, Dict, List
from dotenv import load_dotenv
import requests
from rich.console import Console
from rich.table import Table

# Initialize rich console for better formatting
console = Console()

# Model configurations
MODELS = {
    "llama-3.3-70b-versatile": {
        "id": "llama-3.3-70b-versatile",
        "description": "Current model used in AgentGist",
    },
    "llama3-groq-70b-tool-use": {
        "id": "llama3-groq-70b-tool-use",
        "description": "Specialized for tool use",
    },
}

# Test scenarios
GENERAL_QUESTIONS = [
    "Explain the difference between supervised and unsupervised learning in 2-3 sentences.",
    "What's the most efficient way to analyze large volumes of Reddit posts?",
    "How would you summarize the key points from multiple discussions on a topic?",
]

TOOL_USE_TESTS = [
    {
        "prompt": "Extract the key topics from this Reddit post: 'I've been using Python for data analysis but finding it slow for large datasets. Has anyone tried Julia or Rust for this purpose? What's been your experience with performance improvements?'",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "extract_topics",
                    "description": "Extract key topics from text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of key topics",
                            }
                        },
                        "required": ["topics"],
                    },
                },
            }
        ],
    }
]

STRUCTURED_OUTPUT_TESTS = [
    {
        "prompt": "Analyze this Reddit post and extract the sentiment, main topics, and questions asked: 'I recently upgraded to Windows 11 and I'm experiencing frequent BSODs. Has anyone else encountered this? Are there specific drivers I should update? The crashes seem to happen mostly when using Chrome.'",
        "output_schema": {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                "main_topics": {"type": "array", "items": {"type": "string"}},
                "questions": {"type": "array", "items": {"type": "string"}},
            },
        },
    }
]

# Reddit-specific test for AgentGist
REDDIT_ANALYSIS_TEST = {
    "post": {
        "title": "Is ChatGPT getting worse over time?",
        "content": "I've been using ChatGPT since it first came out, and I feel like the quality of responses has declined in recent months. It seems more prone to hallucinations and misunderstanding my questions. Has anyone else noticed this trend? Are there specific techniques to get better results? Maybe I'm just asking more complex questions now than I was before.",
        "comments": [
            "I've noticed the same thing. It's definitely not as reliable as it used to be.",
            "I think it depends on what you're asking. For creative writing it's still excellent, but for factual information it struggles more than before.",
            "No, I think it's actually improving. You're probably just asking more complex questions that reveal its limitations.",
            "The problem is that they keep changing the models and fine-tuning. Each update seems to fix some issues but create new ones.",
        ],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral", "mixed"]},
            "controversies": {"type": "array", "items": {"type": "string"}},
            "topics": {"type": "array", "items": {"type": "string"}},
        },
    },
}


def call_groq_api(
    model_id: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: int = 1000,
    tools: List[Dict[str, Any]] = None,
    json_response: bool = False,
) -> Dict[str, Any]:
    """Call the Groq API with the given parameters."""
    api_key = os.getenv("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if tools:
        payload["tools"] = tools

    if json_response:
        payload["response_format"] = {"type": "json_object"}

    start_time = time.time()
    response = requests.post(url, json=payload, headers=headers)
    elapsed_time = time.time() - start_time

    if response.status_code == 200:
        result = response.json()
        result["elapsed_time"] = elapsed_time
        return result
    else:
        raise Exception(f"API error: {response.status_code} - {response.text}")


def run_general_question_tests():
    """Test models on general question answering."""
    console.print("\n[bold blue]Running General Question Tests[/bold blue]")

    results = {}

    for model_id, model_info in MODELS.items():
        console.print(f"\n[bold]Testing {model_id}[/bold]")
        model_results = []

        for question in GENERAL_QUESTIONS:
            console.print(f"\nQuestion: {question}")

            try:
                response = call_groq_api(
                    model_id=model_id, messages=[{"role": "user", "content": question}]
                )

                answer = response["choices"][0]["message"]["content"]
                elapsed_time = response.get("elapsed_time", 0)

                console.print(f"Response ({elapsed_time:.2f}s): {answer}")

                model_results.append(
                    {"question": question, "answer": answer, "elapsed_time": elapsed_time}
                )
            except Exception as e:
                console.print(f"[bold red]Error[/bold red]: {str(e)}")
                model_results.append({"question": question, "error": str(e)})

        results[model_id] = model_results

    return results


def run_tool_use_tests():
    """Test models on tool use capabilities."""
    console.print("\n[bold blue]Running Tool Use Tests[/bold blue]")

    results = {}

    for model_id, model_info in MODELS.items():
        console.print(f"\n[bold]Testing {model_id}[/bold]")
        model_results = []

        for test in TOOL_USE_TESTS:
            console.print(f"\nPrompt: {test['prompt']}")

            try:
                response = call_groq_api(
                    model_id=model_id,
                    messages=[{"role": "user", "content": test["prompt"]}],
                    tools=test["tools"],
                )

                message = response["choices"][0]["message"]
                tool_calls = message.get("tool_calls", [])
                content = message.get("content", "")
                elapsed_time = response.get("elapsed_time", 0)

                if tool_calls:
                    tool_call = tool_calls[0]
                    function_name = tool_call["function"]["name"]
                    function_args = json.loads(tool_call["function"]["arguments"])

                    console.print(f"Tool call ({elapsed_time:.2f}s):")
                    console.print(f"Function: {function_name}")
                    console.print(f"Arguments: {json.dumps(function_args, indent=2)}")
                else:
                    console.print(f"No tool call made. Response ({elapsed_time:.2f}s):")
                    console.print(content)

                model_results.append(
                    {
                        "prompt": test["prompt"],
                        "tool_calls": tool_calls,
                        "content": content,
                        "elapsed_time": elapsed_time,
                    }
                )
            except Exception as e:
                console.print(f"[bold red]Error[/bold red]: {str(e)}")
                model_results.append({"prompt": test["prompt"], "error": str(e)})

        results[model_id] = model_results

    return results


def run_structured_output_tests():
    """Test models on structured output formatting."""
    console.print("\n[bold blue]Running Structured Output Tests[/bold blue]")

    results = {}

    for model_id, model_info in MODELS.items():
        console.print(f"\n[bold]Testing {model_id}[/bold]")
        model_results = []

        for test in STRUCTURED_OUTPUT_TESTS:
            console.print(f"\nPrompt: {test['prompt']}")

            try:
                system_message = f"You must respond with a JSON object in this format: {json.dumps(test['output_schema'])}"

                response = call_groq_api(
                    model_id=model_id,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": test["prompt"]},
                    ],
                    json_response=True,
                )

                result_json = response["choices"][0]["message"]["content"]
                elapsed_time = response.get("elapsed_time", 0)

                # Parse the JSON to make sure it's valid
                parsed_json = json.loads(result_json)

                console.print(f"JSON Response ({elapsed_time:.2f}s):")
                console.print(json.dumps(parsed_json, indent=2))

                model_results.append(
                    {"prompt": test["prompt"], "output": parsed_json, "elapsed_time": elapsed_time}
                )
            except json.JSONDecodeError:
                console.print(f"[bold red]Invalid JSON Response[/bold red]: {result_json}")
                model_results.append(
                    {
                        "prompt": test["prompt"],
                        "output": result_json,
                        "error": "Invalid JSON response",
                        "elapsed_time": elapsed_time,
                    }
                )
            except Exception as e:
                console.print(f"[bold red]Error[/bold red]: {str(e)}")
                model_results.append({"prompt": test["prompt"], "error": str(e)})

        results[model_id] = model_results

    return results


def run_reddit_analysis_test():
    """Test models on Reddit post analysis specific to AgentGist."""
    console.print("\n[bold blue]Running AgentGist Reddit Analysis Test[/bold blue]")

    results = {}
    post = REDDIT_ANALYSIS_TEST["post"]

    post_content = f"""
Title: {post['title']}
Content: {post['content']}

Comments:
{chr(10).join(['- ' + comment for comment in post['comments']])}
"""

    prompt = f"Analyze this Reddit post and provide a structured analysis with summary, key points, sentiment, controversies, and topics:\n\n{post_content}"

    for model_id, model_info in MODELS.items():
        console.print(f"\n[bold]Testing {model_id}[/bold]")

        try:
            system_message = f"You must respond with a JSON object in this format: {json.dumps(REDDIT_ANALYSIS_TEST['output_schema'])}"

            response = call_groq_api(
                model_id=model_id,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                json_response=True,
            )

            result_json = response["choices"][0]["message"]["content"]
            elapsed_time = response.get("elapsed_time", 0)

            # Parse the JSON to make sure it's valid
            parsed_json = json.loads(result_json)

            console.print(f"Analysis Response ({elapsed_time:.2f}s):")
            console.print(json.dumps(parsed_json, indent=2))

            results[model_id] = {"analysis": parsed_json, "elapsed_time": elapsed_time}
        except Exception as e:
            console.print(f"[bold red]Error[/bold red]: {str(e)}")
            results[model_id] = {"error": str(e)}

    return results


def display_comparison_table(test_results):
    """Display a comparison table of the results."""
    console.print("\n[bold green]Model Comparison Results[/bold green]")

    table = Table(title="Model Performance Comparison")
    table.add_column("Metric", style="cyan")

    for model_id in MODELS.keys():
        table.add_column(model_id, style="green")

    # General questions performance
    general_results = test_results.get("general", {})
    if general_results:
        avg_times = {}
        for model_id, results in general_results.items():
            times = [r.get("elapsed_time", 0) for r in results if "elapsed_time" in r]
            avg_times[model_id] = sum(times) / len(times) if times else 0

        table.add_row(
            "Avg. Response Time (General)",
            f"{avg_times.get('llama-3.3-70b-versatile', 0):.2f}s",
            f"{avg_times.get('llama3-groq-70b-tool-use', 0):.2f}s",
        )

    # Tool use performance
    tool_results = test_results.get("tool_use", {})
    if tool_results:
        tool_usage = {}
        for model_id, results in tool_results.items():
            tool_calls = [r for r in results if r.get("tool_calls")]
            tool_usage[model_id] = f"{len(tool_calls)}/{len(results)}"

        table.add_row(
            "Tool Usage (Successful/Total)",
            tool_usage.get("llama-3.3-70b-versatile", "0/0"),
            tool_usage.get("llama3-groq-70b-tool-use", "0/0"),
        )

    # Structured output performance
    structured_results = test_results.get("structured", {})
    if structured_results:
        valid_json = {}
        for model_id, results in structured_results.items():
            valid = [r for r in results if "output" in r and "error" not in r]
            valid_json[model_id] = f"{len(valid)}/{len(results)}"

        table.add_row(
            "Valid JSON Responses",
            valid_json.get("llama-3.3-70b-versatile", "0/0"),
            valid_json.get("llama3-groq-70b-tool-use", "0/0"),
        )

    # Reddit analysis performance
    reddit_results = test_results.get("reddit", {})
    if reddit_results:
        analysis_times = {}
        for model_id, result in reddit_results.items():
            if "elapsed_time" in result:
                analysis_times[model_id] = result["elapsed_time"]

        table.add_row(
            "Reddit Analysis Time",
            f"{analysis_times.get('llama-3.3-70b-versatile', 0):.2f}s",
            f"{analysis_times.get('llama3-groq-70b-tool-use', 0):.2f}s",
        )

    console.print(table)


def main():
    """Run all tests and display results."""
    console.print("[bold]Starting Groq Model Comparison Tests[/bold]")
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        console.print(
            "[bold red]Error: GROQ_API_KEY not found in environment variables![/bold red]"
        )
        return

    test_results = {
        "general": run_general_question_tests(),
        "tool_use": run_tool_use_tests(),
        "structured": run_structured_output_tests(),
        "reddit": run_reddit_analysis_test(),
    }

    display_comparison_table(test_results)

    console.print("\n[bold]Test Conclusion[/bold]")
    console.print(
        "Based on the test results, consider which model performs better for AgentGist's specific needs."
    )
    console.print("Key factors to consider:")
    console.print("1. Speed of responses")
    console.print("2. Quality of structured output")
    console.print("3. Tool use capabilities")
    console.print("4. Reddit analysis accuracy")


if __name__ == "__main__":
    main()
