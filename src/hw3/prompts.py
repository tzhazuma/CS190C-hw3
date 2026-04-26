from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step. "
    "Always finish with a final line in the form `#### <answer>`."
)


def build_messages(system_prompt: str, question: str, answer: str | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": question.strip()})
    if answer is not None:
        messages.append({"role": "assistant", "content": answer.strip()})
    return messages


def render_prompt(
    tokenizer,
    *,
    question: str,
    system_prompt: str,
    answer_prefix: str,
    use_chat_template: bool,
    answer: str | None = None,
) -> str:
    can_use_chat_template = bool(use_chat_template and getattr(tokenizer, "chat_template", None))
    if can_use_chat_template:
        messages = build_messages(system_prompt=system_prompt, question=question, answer=answer)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=answer is None,
        )

    prompt_lines: list[str] = []
    if system_prompt.strip():
        prompt_lines.append(system_prompt.strip())
        prompt_lines.append("")
    prompt_lines.append(f"Question: {question.strip()}")
    prompt_lines.append(answer_prefix.rstrip())
    prompt = "\n".join(prompt_lines)
    if answer is None:
        return prompt
    return f"{prompt} {answer.strip()}"


def should_add_special_tokens(tokenizer, use_chat_template: bool) -> bool:
    return not bool(use_chat_template and getattr(tokenizer, "chat_template", None))
