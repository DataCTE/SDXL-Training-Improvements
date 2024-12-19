"""Common type definitions."""
from enum import Enum, auto
import inspect

class DataType(Enum):
    """Data types for model components."""
    FLOAT_32 = auto()
    FLOAT_16 = auto()
    BFLOAT_16 = auto()

class ModelWeightDtypes:
    """Data type configuration for model components."""
    
    def __init__(
        self,
        train_dtype: DataType,
        fallback_train_dtype: DataType,
        unet: DataType,
        prior: DataType,
        text_encoder: DataType,
        text_encoder_2: DataType,
        text_encoder_3: DataType,
        vae: DataType,
        effnet_encoder: DataType,
        decoder: DataType,
        decoder_text_encoder: DataType,
        decoder_vqgan: DataType,
        lora: DataType,
        embedding: DataType,
    ):
        """Initialize model weight data types.
        
        Args:
            train_dtype: Primary training dtype
            fallback_train_dtype: Fallback training dtype
            unet: UNet model dtype
            prior: Prior model dtype
            text_encoder: Text encoder dtype
            text_encoder_2: Second text encoder dtype
            text_encoder_3: Third text encoder dtype
            vae: VAE dtype
            effnet_encoder: EfficientNet encoder dtype
            decoder: Decoder dtype
            decoder_text_encoder: Decoder text encoder dtype
            decoder_vqgan: Decoder VQGAN dtype
            lora: LoRA adapter dtype
            embedding: Embedding dtype
        """
        self.train_dtype = train_dtype
        self.fallback_train_dtype = fallback_train_dtype

        self.unet = unet
        self.prior = prior
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_3 = text_encoder_3
        self.vae = vae
        self.effnet_encoder = effnet_encoder
        self.decoder = decoder
        self.decoder_text_encoder = decoder_text_encoder
        self.decoder_vqgan = decoder_vqgan
        self.lora = lora
        self.embedding = embedding

    def all_dtypes(self) -> list:
        """Get list of all component dtypes."""
        return [
            self.unet,
            self.prior,
            self.text_encoder,
            self.text_encoder_2,
            self.text_encoder_3,
            self.vae,
            self.effnet_encoder,
            self.decoder,
            self.decoder_text_encoder,
            self.decoder_vqgan,
            self.lora,
            self.embedding,
        ]

    @staticmethod
    def from_single_dtype(dtype: DataType) -> "ModelWeightDtypes":
        """Create instance with same dtype for all components.
        
        Args:
            dtype: Data type to use for all components
            
        Returns:
            ModelWeightDtypes instance
        """
        params = [dtype for _ in set(inspect.signature(ModelWeightDtypes).parameters.keys())]
        return ModelWeightDtypes(*params)
