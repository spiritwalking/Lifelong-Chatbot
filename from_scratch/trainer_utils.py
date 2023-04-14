from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding
from typing import List, Union


class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str] = True,
        max_length: int = None,
        input_pad_token_id: int = None,
        label_pad_token_id: int = None,
    ):
        super().__init__(tokenizer, padding, max_length)
        self.input_pad_token_id = input_pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def pad(self, features: List[dict]) -> dict:
        input_features = [{"input_ids": f["input_ids"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        padded_input_features = self.tokenizer.pad(
            input_features, padding=self.padding, max_length=self.max_length, pad_token_id=self.input_pad_token_id
        )
        padded_label_features = self.tokenizer.pad(
            label_features, padding=self.padding, max_length=self.max_length, pad_token_id=self.label_pad_token_id
        )

        # Merge the padded input features and label features
        padded_features = {}
        for key in padded_input_features.keys():
            if key != "input_ids":
                padded_features[key] = padded_input_features[key]
        padded_features["input_ids"] = padded_input_features["input_ids"]
        padded_features["labels"] = padded_label_features["input_ids"]

        return padded_features
