import re

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
import lighteval.tasks.default_prompts as prompt


def prompt_hellaswag(line, task_name: str = None):
    def preprocess(text):
        """Comes from AiHarness"""
        # text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=preprocess(line["activity_label"] + ": " + ctx),
        choices=[" " + preprocess(ending) for ending in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,  # -1 for test
    )

def mmlu_cloze_prompt(line, task_name: str = None):
    """MMLU prompt without choices"""
    topic = line["subject"]
    prompt = f"The following are questions about {topic.replace('_', ' ')}.\nQuestion: "
    prompt += line["question"] + "\nAnswer:"

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {c}" for c in line["choices"]],
        gold_index=int(line["answer"]),
        instruction=f"The following are questions about {topic.replace('_', ' ')}.\n",
    )

def bbh_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query="Question: " + line["input"] + "\nAnswer: ",
        choices=[line["target"]],
        gold_index=0,
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="arc:easy",
        prompt_function=prompt.arc,
        suite=["custom"],
        hf_repo="ai2_arc",
        hf_revision="210d026faf9955653af8916fad021475a3f00453",
        hf_subset="ARC-Easy",
        evaluation_splits=["test"],
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="arc:challenge",
        prompt_function=prompt.arc,
        suite=["custom"],
        hf_repo="ai2_arc",
        hf_revision="210d026faf9955653af8916fad021475a3f00453",
        hf_subset="ARC-Challenge",
        evaluation_splits=["test"],
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="openbook_qa",
        prompt_function=prompt.openbookqa,
        suite=["custom"],
        hf_repo="allenai/openbookqa",
        hf_subset="main",
        hf_revision="388097ea7776314e93a529163e0fea805b8a6454",
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="hellaswag",
        prompt_function=prompt_hellaswag,
        suite=["custom"],
        hf_repo="Rowan/hellaswag",
        hf_subset="default",
        hf_revision="6002345709e0801764318f06bf06ce1e7d1a1fe3",
        trust_dataset=True,
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="winogrande",
        prompt_function=prompt.winogrande,
        suite=["custom"],
        hf_repo="allenai/winogrande",
        hf_subset="winogrande_xl",
        hf_revision="85ac5b5a3b7a930e22d590176e39460400d19e41",
        trust_dataset=True,
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="piqa",
        prompt_function=prompt.piqa_harness,
        suite=["custom"],
        hf_repo="ybisk/piqa",
        hf_subset="plain_text",
        hf_revision="2e8ac2dffd59bac8c3c6714948f4c551a0848bb0",
        trust_dataset=True,
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="trivia_qa",
        prompt_function=prompt.triviaqa,
        suite=["custom"],
        hf_repo="mandarjoshi/trivia_qa",
        hf_subset="rc.nocontext",
        hf_revision="0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        metric=[Metrics.quasi_exact_match_triviaqa],
        generation_size=20,
        trust_dataset=True,
        stop_sequence=["Question:", "Question"],
        few_shots_select="random_sampling_from_train",
    ),
    LightevalTaskConfig(
        name="gsm8k",
        prompt_function=prompt.gsm8k,
        suite=["custom"],
        hf_repo="openai/gsm8k",
        hf_subset="main",
        hf_revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        metric=[Metrics.quasi_exact_match_gsm8k],
        generation_size=256,
        stop_sequence=["Question:", "Question"],
        few_shots_select="random_sampling_from_train",
    ),
    LightevalTaskConfig(
        name="mmlu",
        prompt_function=mmlu_cloze_prompt,
        suite=["custom"],
        hf_repo="cais/mmlu",
        hf_subset="all",
        hf_revision="c30699e8356da336a370243923dbaf21066bb9fe",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        metric=[Metrics.loglikelihood_acc_norm_nospace],
        generation_size=-1,
    ),
]

BBH_TASKS = [
    LightevalTaskConfig(
        name=f"bbh:{subset}",
        prompt_function=bbh_prompt,
        suite=["custom"],
        hf_repo="lighteval/big_bench_hard",
        hf_subset=subset,
        hf_revision="80610173426f05e6f1448f047e2db4840a7dd899",
        metric=[Metrics.exact_match],
        hf_avail_splits=["train"],
        # this is the only split available, obviously not used in training
        evaluation_splits=["train"],
        few_shots_split="train",
        trust_dataset=True,
        stop_sequence=["Question:", "Question"],
    )
    for subset in [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "dyck_languages",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "multistep_arithmetic_two",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting",
    ]
]

TASKS_TABLE.extend(BBH_TASKS)

## MMLU ##
class CustomMMLUEvaluationTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        prompt_function=None,
        hf_repo="lighteval/mmlu",
        hf_subset=None,
        #  metric=[Metrics.loglikelihood_acc_single_token],
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        hf_avail_splits=None,
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        suite=["custom"],
        generation_size=-1,
        stop_sequence=None,
        output_regex=None,
        frozen=False,
    ):
        super().__init__(
            name=name,
            prompt_function=prompt_function,
            suite=suite,
            hf_repo=hf_repo,
            hf_subset=hf_subset,
            metric=metric,
            hf_avail_splits=hf_avail_splits,
            evaluation_splits=evaluation_splits,
            few_shots_split=few_shots_split,
            few_shots_select=few_shots_select,
            generation_size=generation_size,
            stop_sequence=stop_sequence,
            output_regex=output_regex,
            frozen=frozen,
        )


if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))