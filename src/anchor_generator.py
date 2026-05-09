import re
import ollama


class AnchorGenerator:
    _SYSTEM_MESSAGE = """
    You are an expert in political discourse analysis and debate extraction.

    Your task is to identify contested policy issues from collections of political statements and reconstruct structured debate pairs.

    Always follow the requested output format exactly.
    Only use information grounded in the provided text.
    Do not invent arguments or add commentary.
    Be concise, neutral, and analytical.
    """

    def __init__(self, model_name='gemma3', random_seed=42):
        self.model_name = model_name
        self.random_seed = random_seed

    def generate(self, summaries_text: str, topic: str, general=False, temperature=0):
        prompt = self._build_prompt(summaries_text, topic, general)

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "system", "content": self._SYSTEM_MESSAGE}
            ],
            options={"temperature": temperature, 'seed': self.random_seed}
        )

        content = response["message"]["content"]
        pattern = r"Issue:\s*(.*?)\nFor:\s*(.*?)\nAgainst:\s*(.*?)(?=\nIssue:|\Z)"
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            print("Warning: no anchors extracted. Raw response:\n", content)
            return []

        anchors = [
            {"topic": m[0].strip(), "pro": m[1].strip(), "con": m[2].strip()}
            for m in matches
        ]

        return anchors[0] if general else anchors

    def _build_prompt(self, summaries_text: str, topic: str, general: bool) -> str:
        if general:
            return f"""
                You are given a dataset of summaries of political statements about the topic: "{topic}".

                Each summary represents the opinion of one politician.

                Your task is to identify the single most important contested policy issue from the dataset and express it as a structured debate pair.

                Method:
                1. Read all summaries.
                2. Identify the most central policy question or normative dispute.
                3. Group similar viewpoints.
                4. Detect two opposing positions on this issue.
                5. Formulate a neutral issue statement and one "For" and one "Against" argument.

                Constraints:
                - Use ONLY positions supported by the summaries.
                - Do NOT invent arguments.
                - The issue must be a clear political question.
                - "For" and "Against" must represent genuinely opposing positions.

                Writing rules:
                - Issue: 10–18 words
                - For / Against: 20–35 words
                - Neutral analytical tone.
                - Do not mention politicians or parties.

                Output format (strict):

                Issue: <Neutral statement of the main contested issue>
                For: <Position emphasising one set of causes, actors, and solutions>
                Against: <Position emphasising a completely different set of causes, actors, and solutions>

                Generate EXACTLY ONE issue.

                Dataset:
                {summaries_text}
                """

        return f"""
        You are given a dataset of summaries of political statements about the topic: "{topic}".

        Each summary represents the opinion of one politician.

        Your task is to reconstruct the main contested policy issues discussed in the dataset and express them as structured debate pairs.

        Method:
        1. Read all summaries.
        2. Identify recurring policy questions or normative disputes.
        3. Group similar viewpoints.
        4. Detect two opposing positions on each issue.
        5. Formulate a neutral issue statement and one "For" and one "Against" argument.

        Constraints:
        - Use ONLY positions supported by the summaries.
        - Do NOT invent arguments.
        - Issues must be clear political questions.
        - "For" and "Against" must represent genuinely opposing positions.

        Writing rules:
        - Issue: 10–18 words
        - For / Against: 20–35 words
        - Neutral analytical tone.
        - Do not mention politicians or parties.
        - Avoid duplicate issues.

        Output format (strict):

        Issue: <Neutral statement of the contested issue>
        For: <Argument supporting the issue>
        Against: <Argument opposing the issue>

        Issue: <Neutral statement of the contested issue>
        For: <Argument supporting the issue>
        Against: <Argument opposing the issue>

        Generate between 5 and 8 issues.

        Dataset:
        {summaries_text}
        """
