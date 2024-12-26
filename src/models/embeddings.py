"""Base embedding class for SDXL model implementations."""

class BaseModelEmbedding:
    def __init__(
        self,
        uuid: str,
        token_count: int,
        placeholder: str,
    ):
        if not uuid:
            raise ValueError("UUID must not be empty")
        if token_count <= 0:
            raise ValueError("Token count must be positive")

        self.uuid = uuid
        self.token_count = token_count
        self.placeholder = placeholder if placeholder else f"<embedding-{uuid}>"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(uuid='{self.uuid}', token_count={self.token_count})"
