import re

class LLMQuestionGenerator:
    def __init__(self, model, tokenizer, question_template: str, seed: int = 0, max_new_tokens: int = 512,
                 do_sample: bool = True, temperature: float = 1.5):
        """
        Generate questions using a language model, according to a question template as a prompt.
        Args:
            model: The language model. Must have a generate method.
            tokenizer: The tokenizer for the language model. Must have an apply_chat_template method.
            question_template: The question template to use as a prompt.
            seed: The seed for the language model.
            max_new_tokens: The maximum number of tokens to generate.
            do_sample: Parameter for the generate method.
            temperature: Parameter for the generate method.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.question_template = question_template
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature

        self.model.eval()

        self.message = [{"role": "user", "content": question_template}]
        self.prompt_tokens = tokenizer.apply_chat_template(self.message, return_tensors="pt").to(self.model.device)

    def generate(self, n_questions: int, save_path: str = None) -> list[str]:
        """
        Generate questions using the question template.
        Args:
            n_questions (int): The number of questions to generate.
        Returns:
            List[str]: The generated questions.
        """
        if save_path:
            f = open(save_path, "w")

        questions = []
        i = 0
        while i < n_questions:
            generated_ids = self.model.generate(
                self.prompt_tokens,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
            )
            question_raw = self.tokenizer.decode(generated_ids[0].tolist())

            # remove the question template from the generated question
            pattern = r'\[\/INST\]\s*"([^"]*)"\s*<\/s>'
            match = re.search(pattern, question_raw)
            if match:
                question = match.group(1)
            else:
                continue

            # remove all \n and \r characters
            question = question.replace("\n", " ").replace("\r", " ")
            questions.append(question)

            if save_path:
                f.write(question + "\n")
                f.flush()

            i += 1

        f.close()
        return questions


if __name__ == '__main__':
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mistral_models_path = os.path.join(os.path.dirname(__file__), "../../model/llm/")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
                                              cache_dir=mistral_models_path, device_map='auto')
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
                                                 cache_dir=mistral_models_path, device_map='auto')

    question_template = '''Create a summarizable question about adult salaries using the following related information: age, capital-gain, capital-loss, education-num, education, fnlwgt, hours-per-week, marital-status, native-country, occupation, race, relationship, sex, workclass, and salary>50K. For example: "Which education level results in the highest salary?"'''
    question_generator = LLMQuestionGenerator(model, tokenizer, question_template)

    n_questions = 10000
    save_path = os.path.join(os.path.dirname(__file__), "../../data/adult/processed/summary_questions.txt")
    questions = question_generator.generate(n_questions, save_path=save_path)


