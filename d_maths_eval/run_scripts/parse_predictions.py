#!/usr/bin/env python
import re
import logging

def parse_prediction_text(text, allowed_operations):
    """
    Extracts from a prediction string:
      - The first two numbers found.
      - The first occurrence of an allowed operation.
      - The last number in the text (predicted answer).
      - Computes the correct result and determines if the predicted answer is correct.
      
    Returns a dictionary with the extracted pieces.
    """
    numbers_found = re.findall(r"\d+", text)
    ops_found = re.findall(r"[+\-*/÷]", text)

    result_info = {
        "first_num": None,
        "operation": None,
        "second_num": None,
        "predicted_answer": None,
        "computed_answer": None,
        "correct": False,
    }

    if len(numbers_found) < 2:
        return result_info  # Not enough numbers to perform a calculation

    # First two numbers
    first_num_str = numbers_found[0]
    second_num_str = numbers_found[1]

    # Find the first allowed operation symbol
    operation = None
    for op_candidate in ops_found:
        if op_candidate in allowed_operations:
            operation = op_candidate
            break

    predicted_answer_str = numbers_found[-1]

    # Store initial extracted info
    result_info["first_num"] = first_num_str
    result_info["second_num"] = second_num_str
    result_info["operation"] = operation
    result_info["predicted_answer"] = predicted_answer_str

    # Compute the result if an operation is valid
    if operation is not None:
        try:
            num1 = float(first_num_str)
            num2 = float(second_num_str)

            if operation == "+":
                computed_val = num1 + num2
            elif operation == "-":
                computed_val = num1 - num2
            elif operation in ["*", "×"]:
                computed_val = num1 * num2
            elif operation in ["÷", "/"]:
                if abs(num2) < 1e-12:  # Avoid division by zero
                    return result_info
                computed_val = num1 / num2
            else:
                return result_info

            # Convert computed result to string (no decimals if integer)
            if computed_val.is_integer():
                computed_str = str(int(computed_val))
            else:
                computed_str = f"{computed_val:.4f}"  # keeping four decimals

            result_info["computed_answer"] = computed_str

            # Compare predicted answer to computed value (with tolerance for floats)
            if predicted_answer_str == computed_str:
                result_info["correct"] = True
            else:
                try:
                    pred_val = float(predicted_answer_str)
                    if abs(pred_val - computed_val) < 1e-4:
                        result_info["correct"] = True
                except ValueError:
                    pass

        except Exception:
            pass

    return result_info

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Parse a prediction text and extract calculation information."
    )
    parser.add_argument("--text", type=str, required=True,
                        help="Prediction text to parse.")
    parser.add_argument("--allowed_operations", type=str, default="+-*÷",
                        help="String of allowed operation symbols.")
    args = parser.parse_args()

    # Optionally set up a basic logger
    logging.basicConfig(level=logging.INFO)
    result = parse_prediction_text(args.text, args.allowed_operations)
    print("Parsed Result:")
    print(result)
