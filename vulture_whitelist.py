# Vulture whitelist — false positives that should not be flagged.
# Context manager __exit__ signature is required by the protocol.

exc_type  # noqa
exc_val  # noqa
exc_tb  # noqa
module  # noqa
add_special_tokens  # noqa  # tokenizer mock signature compatibility
skip_special_tokens  # noqa  # tokenizer mock signature compatibility
