import verifiers as vf
from datasets import load_dataset


def load_environment():
    ds = load_dataset("openai/gsm8k", "main")

    async def check_ans(completion: list[dict], answer: str) -> float:
        try:
            response = completion[-1]["content"]

            correct_answer = answer.split("####")[-1].strip()
            predicted_answer = response.split("####")[-1].strip()

            return 1.0 if int(predicted_answer) == int(correct_answer) else 0.0

        except ValueError:
            return 0.0

    rubric = vf.Rubric(funcs=[check_ans])

    return vf.SingleTurnEnv(dataset=ds["train"], eval_dataset=ds["test"], rubric=rubric)
