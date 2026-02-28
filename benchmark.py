import time
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
from anthropic import Anthropic

API_KEY = "anthropic-api-key"
MODEL   = "claude-haiku-4-5-20251001"
TRIALS  = 10
WARMUPS = 3

client = Anthropic(api_key=API_KEY)

TASKS = [
    ("user_001", "What is the email and account balance for user ID user_001?"),
    ("user_002", "What is the email and account balance for user ID user_002?"),
    ("user_003", "What is the email and account balance for user ID user_003?"),
    ("user_004", "What is the email and account balance for user ID user_004?"),
    ("user_005", "What is the email and account balance for user ID user_005?"),
    ("user_006", "What is the email and account balance for user ID user_006?"),
    ("user_007", "What is the email and account balance for user ID user_007?"),
    ("user_008", "What is the email and account balance for user ID user_008?"),
    ("user_009", "What is the email and account balance for user ID user_009?"),
    ("user_010", "What is the email and account balance for user ID user_010?"),
    ("user_011", "What is the email and account balance for user ID user_011?"),
    ("user_012", "What is the email and account balance for user ID user_012?"),
    ("user_013", "What is the email and account balance for user ID user_013?"),
    ("user_014", "What is the email and account balance for user ID user_014?"),
    ("user_015", "What is the email and account balance for user ID user_015?"),
    ("user_016", "What is the email and account balance for user ID user_016?"),
    ("user_017", "What is the email and account balance for user ID user_017?"),
    ("user_018", "What is the email and account balance for user ID user_018?"),
    ("user_019", "What is the email and account balance for user ID user_019?"),
    ("user_020", "What is the email and account balance for user ID user_020?"),
    ("user_021", "What is the email and account balance for user ID user_021?"),
    ("user_022", "What is the email and account balance for user ID user_022?"),
    ("user_023", "What is the email and account balance for user ID user_023?"),
    ("user_024", "What is the email and account balance for user ID user_024?"),
    ("user_025", "What is the email and account balance for user ID user_025?"),
    ("user_026", "What is the email and account balance for user ID user_026?"),
    ("user_027", "What is the email and account balance for user ID user_027?"),
    ("user_028", "What is the email and account balance for user ID user_028?"),
    ("user_029", "What is the email and account balance for user ID user_029?"),
    ("user_030", "What is the email and account balance for user ID user_030?"),
    ("user_031", "What is the email and account balance for user ID user_031?"),
    ("user_032", "What is the email and account balance for user ID user_032?"),
    ("user_033", "What is the email and account balance for user ID user_033?"),
    ("user_034", "What is the email and account balance for user ID user_034?"),
    ("user_035", "What is the email and account balance for user ID user_035?"),
    ("user_036", "What is the email and account balance for user ID user_036?"),
    ("user_037", "What is the email and account balance for user ID user_037?"),
    ("user_038", "What is the email and account balance for user ID user_038?"),
    ("user_039", "What is the email and account balance for user ID user_039?"),
    ("user_040", "What is the email and account balance for user ID user_040?"),
    ("user_041", "What is the email and account balance for user ID user_041?"),
    ("user_042", "What is the email and account balance for user ID user_042?"),
    ("user_043", "What is the email and account balance for user ID user_043?"),
    ("user_044", "What is the email and account balance for user ID user_044?"),
    ("user_045", "What is the email and account balance for user ID user_045?"),
    ("user_046", "What is the email and account balance for user ID user_046?"),
    ("user_047", "What is the email and account balance for user ID user_047?"),
    ("user_048", "What is the email and account balance for user ID user_048?"),
    ("user_049", "What is the email and account balance for user ID user_049?"),
    ("user_050", "What is the email and account balance for user ID user_050?"),
]

MOCK_DB = {
    f"user_{str(i).zfill(3)}": {
        "email":   f"user{i}@example.com",
        "balance": round(random.uniform(100, 9999), 2),
        "name":    f"User {i}"
    }
    for i in range(1, 51)
}

TOOLS = [
    {
        "name": "get_user_profile",
        "description": "Retrieves a user's name and email address from the database given their user ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "The user's ID, e.g. 'user_001'"}
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "get_account_balance",
        "description": "Retrieves a user's current account balance given their user ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "The user's ID, e.g. 'user_001'"}
            },
            "required": ["user_id"]
        }
    }
]

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Simulates calling an external API/database."""
    uid = tool_input.get("user_id")
    if uid not in MOCK_DB:
        return json.dumps({"error": "User not found"})
    
    user = MOCK_DB[uid]
    if tool_name == "get_user_profile":
        return json.dumps({"user_id": uid, "name": user["name"], "email": user["email"]})
    elif tool_name == "get_account_balance":
        return json.dumps({"user_id": uid, "balance": user["balance"]})
    return json.dumps({"error": "Unknown tool"})

def run_non_agentic(task_query: str, user_id: str) -> dict:
    """One-shot prompt with all data baked in. No tools."""
    user = MOCK_DB[user_id]
    
    system = (
        "You are a helpful assistant. Answer the user's question using only "
        "the data provided to you. Be concise."
    )
    prompt = (
        f"Database record for {user_id}:\n"
        f"  Name: {user['name']}\n"
        f"  Email: {user['email']}\n"
        f"  Balance: ${user['balance']}\n\n"
        f"Question: {task_query}"
    )
    
    t_start = time.perf_counter()
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = time.perf_counter() - t_start
    
    return {
        "latency":       latency,
        "input_tokens":  response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens":  response.usage.input_tokens + response.usage.output_tokens,
        "answer":        response.content[0].text,
        "tool_calls":    0,
        "llm_turns":     1
    }

def run_agentic(task_query: str) -> dict:
    """Multi-turn agentic loop: Thought → Tool Call → Observation → Answer."""
    messages = [{"role": "user", "content": task_query}]
    
    total_input_tokens  = 0
    total_output_tokens = 0
    tool_calls_made     = 0
    llm_turns           = 0
    
    t_start = time.perf_counter()
    
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system="You are a helpful assistant with access to tools. Use them to answer the question accurately.",
            tools=TOOLS,
            messages=messages
        )
        
        total_input_tokens  += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        llm_turns           += 1
        
        if response.stop_reason == "end_turn":
            final_answer = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_answer = block.text
            latency = time.perf_counter() - t_start
            return {
                "latency":       latency,
                "input_tokens":  total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens":  total_input_tokens + total_output_tokens,
                "answer":        final_answer,
                "tool_calls":    tool_calls_made,
                "llm_turns":     llm_turns
            }
        
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls_made += 1
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result
                    })
            
            messages.append({"role": "user", "content": tool_results})
        else:
            latency = time.perf_counter() - t_start
            return {
                "latency": latency,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "answer": "ERROR",
                "tool_calls": tool_calls_made,
                "llm_turns": llm_turns
            }

def run_benchmark():
    print("=" * 60)
    print("  AGENTIC TAX BENCHMARK")
    print("=" * 60)
    
    print(f"\n[1/3] Running {WARMUPS} warm-up calls...")
    for i in range(WARMUPS):
        run_non_agentic(TASKS[0][1], TASKS[0][0])
        run_agentic(TASKS[0][1])
        print(f"  Warm-up {i+1}/{WARMUPS} done")
    
    nab_results = []
    arl_results = []
    
    print(f"\n[2/3] Running NON-AGENTIC baseline ({len(TASKS)} tasks × {TRIALS} trials)...")
    for task_idx, (uid, query) in enumerate(TASKS):
        for trial in range(TRIALS):
            result = run_non_agentic(query, uid)
            result["task"] = task_idx + 1
            result["trial"] = trial + 1
            nab_results.append(result)
        print(f"  Task {task_idx+1:02d}/50 complete — avg latency: "
              f"{sum(r['latency'] for r in nab_results[-TRIALS:]) / TRIALS:.2f}s")
    
    print(f"\n[3/3] Running AGENTIC loop ({len(TASKS)} tasks × {TRIALS} trials)...")
    for task_idx, (uid, query) in enumerate(TASKS):
        for trial in range(TRIALS):
            result = run_agentic(query)
            result["task"] = task_idx + 1
            result["trial"] = trial + 1
            arl_results.append(result)
        print(f"  Task {task_idx+1:02d}/50 complete — avg latency: "
              f"{sum(r['latency'] for r in arl_results[-TRIALS:]) / TRIALS:.2f}s")
    
    return pd.DataFrame(nab_results), pd.DataFrame(arl_results)

def analyze_and_plot(nab_df: pd.DataFrame, arl_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    
    metrics = {
        "Avg Latency (s)":       ("latency",      nab_df["latency"].mean(),      arl_df["latency"].mean()),
        "Avg Total Tokens":      ("total_tokens",  nab_df["total_tokens"].mean(), arl_df["total_tokens"].mean()),
        "Avg LLM Turns":         ("llm_turns",     nab_df["llm_turns"].mean(),    arl_df["llm_turns"].mean()),
        "Avg Tool Calls":        ("tool_calls",    nab_df["tool_calls"].mean(),   arl_df["tool_calls"].mean()),
    }
    
    print(f"\n{'Metric':<25} {'NAB':>10} {'ARL':>10} {'Tax':>10}")
    print("-" * 58)
    for label, (_, nab_val, arl_val) in metrics.items():
        tax = f"+{((arl_val - nab_val) / nab_val * 100):.1f}%"
        print(f"{label:<25} {nab_val:>10.2f} {arl_val:>10.2f} {tax:>10}")
    
    token_ratio = arl_df["total_tokens"].mean() / nab_df["total_tokens"].mean()
    print(f"\nToken Overhead Ratio (R_token) = {token_ratio:.2f}x")
    
    nab_df.to_csv("nab_results.csv", index=False)
    arl_df.to_csv("arl_results.csv", index=False)
    print("\nRaw data saved to nab_results.csv and arl_results.csv")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Agentic Tax Benchmark Results", fontsize=14, fontweight="bold")
    
    axes[0].hist(nab_df["latency"], bins=30, alpha=0.7, label="NAB", color="#3498db")
    axes[0].hist(arl_df["latency"], bins=30, alpha=0.7, label="ARL", color="#e74c3c")
    axes[0].set_xlabel("Latency (seconds)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("End-to-End Latency Distribution")
    axes[0].legend()
    
    categories = ["Input Tokens", "Output Tokens", "Total Tokens"]
    nab_vals = [nab_df["input_tokens"].mean(), nab_df["output_tokens"].mean(), nab_df["total_tokens"].mean()]
    arl_vals = [arl_df["input_tokens"].mean(), arl_df["output_tokens"].mean(), arl_df["total_tokens"].mean()]
    
    x = range(len(categories))
    axes[1].bar([i - 0.2 for i in x], nab_vals, 0.4, label="NAB", color="#3498db")
    axes[1].bar([i + 0.2 for i in x], arl_vals, 0.4, label="ARL", color="#e74c3c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, rotation=10)
    axes[1].set_ylabel("Avg Token Count")
    axes[1].set_title("Token Usage Comparison")
    axes[1].legend()
    
    nab_per_task = nab_df.groupby("task")["latency"].mean()
    arl_per_task = arl_df.groupby("task")["latency"].mean()
    
    axes[2].plot(nab_per_task.index, nab_per_task.values, label="NAB", color="#3498db", linewidth=1.5)
    axes[2].plot(arl_per_task.index, arl_per_task.values, label="ARL", color="#e74c3c", linewidth=1.5)
    axes[2].fill_between(nab_per_task.index, nab_per_task.values, arl_per_task.values, alpha=0.15, color="#e74c3c", label="Agentic Tax")
    axes[2].set_xlabel("Task Number")
    axes[2].set_ylabel("Avg Latency (s)")
    axes[2].set_title("Latency per Task (Agentic Tax Shaded)")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=150, bbox_inches="tight")
    print("Chart saved to benchmark_results.png")
    plt.show()

if __name__ == "__main__":
    nab_df, arl_df = run_benchmark()
    analyze_and_plot(nab_df, arl_df)