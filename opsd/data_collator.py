import torch


class SelfDistillationDataCollator:
    """
    Data collator for self-distillation that creates both student and teacher inputs.

    Student: sees only the problem (with chat template)
    Teacher: sees problem + solution + transition prompt (with chat template)

    To enable batch-level operations (like original GKD), we pad prompts to the same length
    within each batch, and track the actual (unpadded) prompt lengths for loss masking.
    """

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        reason_first=True,
        student_thinking=False,
        teacher_thinking=True,
        hint_level="L1",
        shared_bridge=False,
        routed_bridge=False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking
        self.hint_level = hint_level
        self.shared_bridge = shared_bridge
        self.routed_bridge = routed_bridge

        supported_hint_levels = {
            "L1", "L2", "L3", "L4", "L5", "UNIFORM", "MIXED"
        }
        if hint_level not in supported_hint_levels:
            raise ValueError(
                f"Unsupported hint_level={hint_level!r}; "
                f"expected one of {sorted(supported_hint_levels)}"
            )
        if shared_bridge and routed_bridge:
            raise ValueError("shared_bridge and routed_bridge are mutually exclusive")
        if hint_level in {"UNIFORM", "MIXED"} and not (
            shared_bridge or routed_bridge
        ):
            raise ValueError(
                f"hint_level={hint_level!r} requires either shared_bridge=True or "
                "routed_bridge=True"
            )
        if routed_bridge and hint_level not in {"UNIFORM", "MIXED"}:
            raise ValueError(
                "routed_bridge=True is only valid for UNIFORM or MIXED datasets "
                "with a per-row hint_level_source field"
            )
        if routed_bridge and reason_first:
            raise ValueError(
                "routed_bridge=True currently requires reason_first=False so each "
                "row uses its exact level-specific teacher transition"
            )

        # ── Level-specific labels for the teacher prompt ──
        self.hint_labels = {
            "L1": ("reference solution",
                "=== Reference Solution Begin ===",
                "=== Reference Solution End ==="),
            "L2": ("strategy hint",
                "=== Strategy Hint Begin ===",
                "=== Strategy Hint End ==="),
            "L3": ("thinking direction",
                "=== Thinking Direction Begin ===",
                "=== Thinking Direction End ==="),
            "L4": ("problem category",   # 去掉 " hint" 后缀，与 L1 保持"名词短语"一致
                "=== Problem Category Begin ===",
                "=== Problem Category End ==="),
            "L5": ("known final answer",
                "=== Final Answer Begin ===",
                "=== Final Answer End ==="),
        }
        if routed_bridge:
            # Selected per row from hint_level_source in __call__.
            self.hint_label = None
            self.hint_intro = None
            self.hint_begin = None
            self.hint_end = None
        elif shared_bridge:
            self.hint_label = "additional information"
            self.hint_intro = "Here is additional information for this problem:"
            self.hint_begin = "=== Additional Information Begin ==="
            self.hint_end = "=== Additional Information End ==="
        else:
            label, begin, end = self.hint_labels[hint_level]
            self.hint_label = label
            self.hint_intro = f"Here is a {label} for this problem:"
            self.hint_begin = begin
            self.hint_end = end

        # ── Level-specific transition prompts ──
        self.transition_prompts = {
            "L1": (
                "\n\nAfter reading the reference solution above, make sure you truly understand "
                "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
                "own words and independent reasoning, derive the same final answer to the problem above. "
                "Think step by step, explore different approaches, and don't be afraid to backtrack "
                "or reconsider if something doesn't work out:\n"
            ),
            "L2": (
                "\n\nAfter reading the strategy hint above, make sure you truly understand "
                "why this strategy fits the problem — do not copy or paraphrase it. Now, using your "
                "own words and independent reasoning, expand this strategy into a full derivation "
                "and arrive at the final answer to the problem above. "
                "Think step by step, explore different approaches, and don't be afraid to backtrack "
                "or reconsider if something doesn't work out:\n"
            ),
            "L3": (
                "\n\nAfter reading the thinking direction above, make sure you truly grasp "
                "how this direction applies to the problem — do not copy or paraphrase it. Now, using your "
                "own words and independent reasoning, discover a concrete method along this direction "
                "and arrive at the final answer to the problem above. "
                "Think step by step, explore different approaches, and don't be afraid to backtrack "
                "or reconsider if something doesn't work out:\n"
            ),
            "L4": (
                "\n\nAfter reading the problem category above, make sure you truly recall "
                "the standard techniques used for this type of problem — do not copy or paraphrase it. Now, using your "
                "own words and independent reasoning, select a suitable technique and derive "
                "the final answer to the problem above. "
                "Think step by step, explore different approaches, and don't be afraid to backtrack "
                "or reconsider if something doesn't work out:\n"
            ),
            "L5": (
                "\n\nThe known final answer above gives only the destination, not the derivation. "
                "Do not merely repeat or cite it. Now independently construct a complete, valid "
                "reasoning path from the problem to that answer, justifying every necessary step. "
                "Think step by step, explore different approaches, and don't be afraid to backtrack "
                "or reconsider if something doesn't work out:\n"
            ),
        }
        if routed_bridge:
            # Selected per row from hint_level_source in __call__.
            self.transition_prompt = None
        elif shared_bridge:
            self.transition_prompt = (
                "\n\nUse the additional information above to independently solve the problem. "
                "Do not merely copy or cite it. Reason step by step, justify the necessary "
                "steps, and put the final answer within \\boxed{}.\n"
            )
        else:
            self.transition_prompt = self.transition_prompts[hint_level]

        # Prompt for reasoning about the solution before teaching
        if hint_level == "L5":
            self.reason_first_prompt = (
                "\n\nThe final answer above is correct but provides no reasoning. "
                "Please analyze the problem and construct a valid derivation that reaches it. "
                "Do NOT merely restate the answer and do NOT use <think> tags.\n"
            )
        else:
            self.reason_first_prompt = (
                "\n\nThe reference reasoning above arrives at the correct answer. "
                "Please analyze this solution and explain the key reasoning steps and problem-solving strategies employed. "
                "Do NOT use <think> tags. Do NOT derive your own solution. "
                "Simply analyze and explain the reference solution provided above.\n"
            )

        print(f"[DataCollator] Hint level: {hint_level}")
        print(f"[DataCollator] Shared bridge: {shared_bridge}")
        print(f"[DataCollator] Routed bridge: {routed_bridge}")

        # Set padding side explicitly for consistency
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")

    def _prompt_components(self, feature):
        """Return the exact teacher wrapper and transition for one row."""
        if self.routed_bridge:
            if "hint_level_source" not in feature:
                raise KeyError(
                    "Routed mixture row is missing required field "
                    "'hint_level_source'. Ensure the parquet contains routing "
                    "metadata and Trainer remove_unused_columns is disabled."
                )
            row_level = str(feature["hint_level_source"]).strip()
            if row_level not in self.hint_labels:
                raise ValueError(
                    f"Unsupported hint_level_source={row_level!r}; expected one "
                    f"of {sorted(self.hint_labels)}"
                )
            label, hint_begin, hint_end = self.hint_labels[row_level]
            return (
                row_level,
                f"Here is a {label} for this problem:",
                hint_begin,
                hint_end,
                self.transition_prompts[row_level],
            )

        return (
            self.hint_level,
            self.hint_intro,
            self.hint_begin,
            self.hint_end,
            self.transition_prompt,
        )

    def __call__(self, features):

        batch_size = len(features)

        # Prepare student and teacher prompts using chat template (matching evaluation)
        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []  # NEW: for reason_first mode

        for feature in features:
            # Extract problem and solution from dataset
            # Handle different possible column names
            problem = feature["problem"]
            solution = feature["solution"]
            (
                row_level,
                hint_intro,
                hint_begin,
                hint_end,
                transition_prompt,
            ) = self._prompt_components(feature)

            # Student prompt: just the problem with instruction (matching evaluation format)
            student_user_message = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            student_messages = [{"role": "user", "content": student_user_message}]

            # Apply chat template for student (matching evaluation)
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.student_thinking
            )
            student_prompts.append(student_prompt)

            if self.reason_first:
                # Reasoning prompt: ask teacher to analyze the solution
                reasoning_user_message = (
                    f"Problem: {problem}\n\n"
                    f"{hint_intro}\n"
                    f"{hint_begin}\n"
                    f"{solution}\n"
                    f"{hint_end}\n\n"
                    f"{self.reason_first_prompt}"
                )
                reasoning_messages = [{"role": "user", "content": reasoning_user_message}]
                reasoning_prompt = self.tokenizer.apply_chat_template(
                    reasoning_messages, tokenize=False, add_generation_prompt=True
                )
                teacher_reasoning_prompts.append(reasoning_prompt)

                # Teacher prompt will be constructed during training after reasoning
                # For now, create placeholder (will be replaced in training_step)
                teacher_prompts.append("")  # Placeholder
            else:
                # Teacher prompt with level-specific hint label and transition
                teacher_user_message = (
                    f"Problem: {problem}\n\n"
                    f"{hint_intro}\n"
                    f"{hint_begin}\n{solution}\n{hint_end}\n"
                    f"{transition_prompt}"
                )
                teacher_messages = [{"role": "user", "content": teacher_user_message}]

                # Apply chat template for teacher
                teacher_prompt = self.tokenizer.apply_chat_template(
                    teacher_messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.teacher_thinking
                )
                teacher_prompts.append(teacher_prompt)

        # Tokenize WITHOUT padding first to get true lengths
        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]

        # Find max lengths in this batch
        max_student_prompt_len = max(student_prompt_lengths)

        # Tokenize WITH padding to max length in batch
        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,  # Single value for batch!
            # Keep individual lengths for proper masking
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
        }

        if self.reason_first:
            # Tokenize reasoning prompts
            reasoning_encoded_no_pad = self.tokenizer(
                teacher_reasoning_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
            max_reasoning_prompt_len = max(reasoning_prompt_lengths)

            reasoning_encoded = self.tokenizer(
                teacher_reasoning_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reasoning_prompt_len,
                return_tensors="pt",
            )

            # Tokenize transition prompt (this will be appended after reasoning)
            # Don't use chat template here - just the raw text
            transition_text = f"\n{self.transition_prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            transition_encoded = self.tokenizer(
                [transition_text] * batch_size,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
            # Normal mode: tokenize teacher prompts
            teacher_encoded_no_pad = self.tokenizer(
                teacher_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
            max_teacher_prompt_len = max(teacher_prompt_lengths)

            teacher_encoded = self.tokenizer(
                teacher_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_teacher_prompt_len,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_prompts": teacher_encoded["input_ids"],
                    "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                }
            )

        return result
