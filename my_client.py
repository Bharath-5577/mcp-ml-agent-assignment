
# my_client.py
# Custom MCP client for comparing Logistic Regression and KNN

import asyncio
import json
from pathlib import Path

from agents.mcp import MCPServerStdio


PROJECT_DIR = Path(__file__).resolve().parent

# MCP server configuration
params = {
    "command": "uv",
    "args": ["run", "python", "mcp_ml_server.py"],
    "cwd": str(PROJECT_DIR),
}


async def call(server, tool_name, arguments):
    """Call an MCP tool and return its response as a dictionary."""

    response = await server.call_tool(tool_name, arguments)

    structured = getattr(response, "structuredContent", None)

    if isinstance(structured, dict):
        if (
            set(structured) == {"result"}
            and isinstance(structured["result"], dict)
        ):
            return structured["result"]

        return structured

    for item in response.content:
        if hasattr(item, "text"):
            return json.loads(item.text)

    raise ValueError(f"No valid response from {tool_name}")


async def main():

    async with MCPServerStdio(
        params=params,
        client_session_timeout_seconds=120
    ) as server:

        # STEP 1: Discover MCP tools
        tools = await server.list_tools()

        print(f"\nThe server offers {len(tools)} tools:")

        for tool in tools:
            print(" -", tool.name)

        if not any(tool.name == "predict_both" for tool in tools):
            raise RuntimeError(
                "predict_both is missing from mcp_ml_server.py"
            )

        # STEP 2: Train both ML models
        print("\nTraining Logistic Regression and KNN...")

        training = await call(
            server,
            "train_models",
            {"request_id": "hw-train"}
        )

        # Accept both valid training success statuses
        if training.get("status") not in (
            "success",
            "trained_and_exported"
        ):
            raise RuntimeError(
                f"Training failed: {training}"
            )

        print("Training completed successfully!")

        # STEP 3: Compare predictions for 114 test patients
        disagreements = []
        lr_correct = 0
        knn_correct = 0

        print("\nComparing predictions for 114 test patients...")
        print("-" * 60)

        for i in range(114):

            # Get patient data
            sample = await call(
                server,
                "get_sample_input",
                {
                    "index": i,
                    "request_id": f"hw-sample-{i}"
                }
            )

            if sample.get("status") != "success":
                raise RuntimeError(
                    f"Failed to get patient {i}: {sample}"
                )

            # Predict using both models
            result = await call(
                server,
                "predict_both",
                {
                    "input_data": sample["input_data"],
                    "request_id": f"hw-both-{i}"
                }
            )

            if result.get("status") != "success":
                raise RuntimeError(
                    f"Prediction failed for patient {i}: {result}"
                )

            # Identify disagreements
            if not result["models_agree"]:

                disagreements.append(i)

                actual = sample["true_class"]
                lr = result["logistic_regression"]
                knn = result["knn"]

                lr_right = lr == actual
                knn_right = knn == actual

                lr_correct += int(lr_right)
                knn_correct += int(knn_right)

                if lr_right:
                    winner = "Logistic Regression"
                elif knn_right:
                    winner = "KNN"
                else:
                    winner = "Neither"

                print(
                    f"Patient {i}: "
                    f"Actual={actual}, "
                    f"LR={lr}, "
                    f"KNN={knn}, "
                    f"Correct={winner}"
                )

        # STEP 4: Print final results
        print("\n" + "=" * 60)
        print("FINAL ASSIGNMENT RESULTS")
        print("=" * 60)

        print("Total test patients: 114")
        print(f"Total disagreements: {len(disagreements)}")
        print(f"Disagreement patient indices: {disagreements}")

        print(
            f"Logistic Regression correct on disagreements: "
            f"{lr_correct}"
        )

        print(
            f"KNN correct on disagreements: "
            f"{knn_correct}"
        )

        if lr_correct > knn_correct:
            print(
                "Better model on disagreements: "
                "Logistic Regression"
            )

        elif knn_correct > lr_correct:
            print(
                "Better model on disagreements: KNN"
            )

        else:
            print(
                "Both models performed equally on disagreements"
            )

        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
