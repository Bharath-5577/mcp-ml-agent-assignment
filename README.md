# MCP ML Agent — Custom Tool Assignment

## Project Overview

This project demonstrates how to build and call a custom Model Context Protocol (MCP) tool using Python.

The MCP server provides machine learning tools for training, predicting, evaluating, and auditing Logistic Regression and K-Nearest Neighbors (KNN) models.

For this assignment, I implemented a custom tool called `predict_both`, which compares predictions from both models for the same patient.

## Technologies Used

- Python
- Model Context Protocol (MCP)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Scikit-learn
- Parquet
- uv package manager

## Custom MCP Tool: predict_both

The `predict_both` tool:

1. Accepts a patient's input features.
2. Generates predictions using Logistic Regression and KNN.
3. Returns both predictions.
4. Indicates whether the models agree.
5. Records the tool execution for auditing.

## How to Run

Clone the repository:

```bash
git clone https://github.com/Bharath-5577/mcp-ml-agent-assignment.git
cd mcp-ml-agent-assignment
```

Install dependencies:

```bash
uv sync
```

Run the custom client:

```bash
uv run python my_client.py
```

## Experiment Results

The custom client evaluated 114 test patients.

| Metric | Result |
|---|---|
| Total test patients | 114 |
| Model disagreements | 8 |
| Logistic Regression correct on disagreements | 7 |
| KNN correct on disagreements | 1 |
| Better model on disagreement cases | Logistic Regression |

The disagreement patient indices were:

`[3, 25, 48, 53, 76, 80, 99, 102]`

These results show that Logistic Regression performed better on the eight disagreement cases. This does not necessarily establish that it performs better across every evaluation metric.

## Project Files

- `mcp_ml_server.py` — MCP server and custom tool
- `my_client.py` — Client that calls the MCP tools
- `answers.md` — Written assignment responses
- `screenshots/` — Execution evidence
- `uv.lock` — Dependency lockfile

## Conclusion

This assignment helped me understand how to extend an existing MCP server, expose a custom tool, call it from an MCP client, compare machine learning predictions, and maintain an audit trail.

The experiment demonstrated how a custom MCP tool can simplify model comparison and make machine learning workflows easier to inspect.